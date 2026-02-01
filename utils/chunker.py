from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

MAX_CHARS_PER_CHUNK = 3500
FINAL_SUMMARY_SENTENCES = 5

embeddings = LocalHFEmbeddings()

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)


def chunk_text(text, max_char_limit=MAX_CHARS_PER_CHUNK):
    # paragraphs = text.split("\n\n")
    # chunks = []
    # current_chunk = ""

    # for para in paragraphs:
    #     if len(current_chunk) + len(para) <= max_char_limit:
    #         current_chunk += para + "\n\n"
    #     else:
    #         chunks.append(current_chunk.strip())
    #         current_chunk = para + "\n\n"

    # if current_chunk.strip():
    #     chunks.append(current_chunk.strip())

    # print(chunks)

    semantic_chunks = text_splitter.split_text(text)
    return semantic_chunks