"""本地多语言向量模型 bge-m3（int8 ONNX）：中英文共用一个向量空间，不依赖外部服务。

ChromaDB 自带的 all-MiniLM-L6-v2 词表不认识大部分中文字，中文会变成 [UNK]，因此单独加载模型。
换模型等于换向量空间：新向量写入带模型名的新集合，旧集合保留不动，首次打开时迁移一次。

命令行预下载（Docker 构建使用）：python core/embedding.py
"""
import hashlib
import logging
import os
import pathlib
import threading
import urllib.request

logger = logging.getLogger(__name__)

NAME = "bge-m3-int8"
REPO = "Xenova/bge-m3"
REVISION = "4de13258303883538bd53b696b452bf8099f0858"
FILES = {
    "model.onnx": ("onnx/model_int8.onnx", "a206e10e995aa2a833924bcd725ba5dd6c3425cd34bac3cf2b5677cd2a1c51d6"),
    "tokenizer.json": ("tokenizer.json", "6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790"),
}
MAX_TOKENS = 512

_lock = threading.Lock()
_current = None


def model_dir() -> pathlib.Path:
    return pathlib.Path(os.getenv("KORA_EMBEDDING_DIR") or "~/.cache/kora/models/bge-m3").expanduser()


def ensure_files(directory: pathlib.Path, download: bool = True) -> None:
    """补齐模型文件。固定版本并校验 SHA-256，校验通过才改名，不会留下半个文件。"""
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    for name, (remote, expected) in FILES.items():
        target = directory / name
        if target.exists():
            continue
        if not download:
            raise RuntimeError(f"缺少向量模型文件 {target}；请先运行 python core/embedding.py，"
                               "或设置 KORA_EMBEDDING_PROVIDER=chroma 使用旧模型")
        directory.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                _download(f"{endpoint}/{REPO}/resolve/{REVISION}/{remote}", target, expected)
                break
            except OSError as ex:       # 网络错误重试；校验失败不重试
                if attempt == 2:
                    raise
                logger.warning("下载 %s 失败，重试：%s", name, type(ex).__name__)


def _download(url: str, target: pathlib.Path, expected: str) -> None:
    partial, digest = target.with_name(target.name + ".part"), hashlib.sha256()
    logger.info("下载向量模型 %s", target.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            while block := response.read(1 << 20):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"向量模型 {target.name} 校验失败")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


class LocalEmbedder:
    """线程安全；输出已归一化，点积即余弦相似度。"""

    name = NAME

    def __init__(self, directory: pathlib.Path):
        import numpy
        import onnxruntime
        from tokenizers import Tokenizer
        self._np = numpy
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.enable_truncation(MAX_TOKENS)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, int(os.getenv("KORA_EMBEDDING_THREADS", "2")))
        # 内存池会长期占住一次长文本用过的峰值内存。
        options.enable_cpu_mem_arena = options.enable_mem_pattern = False
        self._session = onnxruntime.InferenceSession(str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"])

    def embed(self, texts):
        np, vectors = self._np, []
        for text in texts:      # 逐条计算：没有填充浪费，实测比成批更快、更省内存
            ids = np.array([self._tokenizer.encode(text).ids], dtype=np.int64)
            hidden = self._session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0]
            vector = hidden[0, 0]
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors


def load_embedder():
    """按配置加载并缓存；KORA_EMBEDDING_PROVIDER=chroma 时返回 None，沿用 ChromaDB 默认模型。"""
    global _current
    provider = os.getenv("KORA_EMBEDDING_PROVIDER", "bge-m3").strip().lower()
    if provider not in {"bge-m3", "chroma"}:
        raise ValueError("KORA_EMBEDDING_PROVIDER 只支持 bge-m3/chroma")
    with _lock:
        if provider == "chroma":
            _current = None
        elif _current is None:
            production = os.getenv("KORA_ENV", "development").lower() == "production"
            allowed = os.getenv("KORA_EMBEDDING_DOWNLOAD", "false" if production else "true").lower() in {"1", "true", "yes"}
            directory = model_dir()
            # 不静默退回旧模型：那会让中文检索大面积失效，且写入另一个集合。
            ensure_files(directory, allowed)
            _current = LocalEmbedder(directory)
            logger.info("向量模型已加载: %s", NAME)
        return _current


def current():
    """已加载的模型；未加载时为 None，不触发加载。"""
    return _current


class EmbeddedCollection:
    """给 ChromaDB 集合补上向量计算，其余方法原样透传。"""

    def __init__(self, collection, embedder, text_of=None):
        self._collection, self._embedder = collection, embedder
        self._text_of = text_of or (lambda document, metadata: document)

    def __getattr__(self, name):
        return getattr(self._collection, name)

    def _vectors(self, documents, metadatas):
        metadatas = metadatas or [None] * len(documents)
        return self._embedder.embed([self._text_of(d, m or {}) for d, m in zip(documents, metadatas)])

    def upsert(self, *, ids, documents, metadatas=None):
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=self._vectors(documents, metadatas))

    def add(self, *, ids, documents, metadatas=None):
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=self._vectors(documents, metadatas))

    def query(self, *, query_texts, **kwargs):
        return self._collection.query(query_embeddings=self._embedder.embed(query_texts), **kwargs)


def open_collection(client, name, embedder, *, text_of=None, metadata=None):
    """打开集合。有本地模型时使用独立的余弦集合，并从旧集合迁移一次。"""
    if embedder is None:
        return client.get_or_create_collection(name=name, **({"metadata": metadata} if metadata else {}))
    target_name = f"{name}.{embedder.name}"
    raw = client.get_or_create_collection(name=target_name, embedding_function=None,
                                          metadata={**(metadata or {}), "hnsw:space": "cosine"})
    target = EmbeddedCollection(raw, embedder, text_of)
    _migrate(client, name, target_name + ".ready", target)
    return target


def _migrate(client, source_name, ready_name, target, batch=64):
    """重新计算旧集合的向量。用标记集合记录完成：之后删除的内容不会从旧集合复活，中断则下次重来。"""
    # 列举失败（如连接断开）必须抛出，不能当成“旧集合不存在”。
    names = {item if isinstance(item, str) else item.name for item in client.list_collections()}
    if ready_name in names:
        return 0
    moved = 0
    if source_name in names:
        source = client.get_collection(name=source_name, embedding_function=None)
        total = source.count()
        if total:
            logger.info("正在用 %s 重建 %s 的向量，共 %s 条", target._embedder.name, source_name, total)
        for offset in range(0, total, batch):
            rows = source.get(limit=batch, offset=offset, include=["documents", "metadatas"])
            kept = [(i, d, m or {"migrated": True}) for i, d, m in zip(rows["ids"], rows["documents"], rows["metadatas"]) if d]
            if kept:
                ids, documents, metadatas = map(list, zip(*kept))
                target.upsert(ids=ids, documents=documents, metadatas=metadatas)
                moved += len(kept)
    client.get_or_create_collection(name=ready_name, embedding_function=None)
    if moved:
        logger.info("%s 迁移完成：%s 条，旧集合已保留", source_name, moved)
    return moved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_files(model_dir())
    print("向量模型已就绪:", model_dir())
