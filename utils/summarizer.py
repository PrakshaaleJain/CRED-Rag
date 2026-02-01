from chunker import *


def summarize_chunk(chunk_text):
    prompt = f"""
#     Write ONE sentence of 35–36 words summarizing the company described below.

# Rules:
# - Use ONLY facts explicitly stated in the text
# - Identify the company’s primary products or services
# - Indicate the main industry or sector only if clearly stated or directly implied
# - Mention customers or distribution channels if described
# - If multiple activities exist, focus on the core revenue-generating business

# Strictly avoid:
# - Guessing or inferring industries, services, or business models
# - Adding software, digital, AI, marketing, or technology unless explicitly described
# - Generic filler language or assumptions

# If the text does not clearly state an industry, describe what the company produces or sells without labeling an industry.
# Before writing the sentence, verify that every product or service you mention appears verbatim or near-verbatim in the text.
# If any item does not appear in the text, exclude it.


# Company description:
# {chunk_text}

"""
    return generate_local(prompt, max_new_tokens=120)


def summarize_whole_text(text):
    chunks = chunk_text(text)
    chunk_summaries = []

    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}")
        summary = summarize_chunk(chunk)
        chunk_summaries.append(summary)

    combined_summary_text = "\n".join(chunk_summaries)

    final_prompt = f"""
    Write ONE sentence of 35–36 words summarizing the company described below.

Rules:
- Use ONLY facts explicitly stated in the text
- Identify the company’s primary products or services
- Indicate the main industry or sector only if clearly stated or directly implied
- Mention customers or distribution channels if described
- If multiple activities exist, focus on the core revenue-generating business

Strictly avoid:
- Guessing or inferring industries, services, or business models
- Adding software, digital, AI, marketing, or technology unless explicitly described
- Generic filler language or assumptions

If the text does not clearly state an industry, describe what the company produces or sells without labeling an industry.
Before writing the sentence, verify that every product or service you mention appears verbatim or near-verbatim in the text.
If any item does not appear in the text, exclude it.


Company description:
{combined_summary_text}

    """

    return generate_local(final_prompt, max_new_tokens=120)

