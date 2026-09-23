def extract_facts(text, generate_fn, max_facts=15):
    """
    generate_fn: a function that takes a prompt string and returns a text reply.
        Example for your local Brain:
            generate_fn = lambda prompt: brain.think([{"role": "user", "content": prompt}])["text"]
        Example for Gemini:
            generate_fn = lambda prompt: gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            ).text
    """
    prompt = (
        f"Extract up to {max_facts} short, atomic facts from the text below.\n"
        "One fact per line. No numbering. No extra commentary.\n\n"
        f"{text[:6000]}"
    )
    reply = generate_fn(prompt)
    facts = [line.strip("-• ").strip() for line in reply.split("\n")]
    return [f for f in facts if f]


# FIX: chunk_size bumped from 3000 -> 8000. This was the main hidden time sink:
# a long conversation transcript was being cut into many 3000-char pieces,
# and EVERY piece triggered a full LLM generation call (through generate_fn),
# none of which was ever timed in your results CSV. Bigger chunks = far
# fewer generation calls for the same transcript, with only a small drop in
# per-chunk fact granularity.
def extract_facts_long(text, generate_fn, chunk_size=8000, max_facts_per_chunk=10):
    """Splits long text into chunks so no single call exceeds the model's context.
    Loops through the WHOLE text instead of cutting it off."""
    all_facts = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        facts = extract_facts(chunk, generate_fn, max_facts=max_facts_per_chunk)
        all_facts.extend(facts)
    return all_facts