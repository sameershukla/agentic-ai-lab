# -----------------------------------------
# Why Chunking Matters: Three Failure Modes
# -----------------------------------------

text = (
    "Adult patients should take paracetamol at a dose of 500mg to 1000mg every four to six hours. "
    "Maximum daily dose is 4000mg. "
    "For wounds, clean with saline and apply a sterile dressing. "
    "Controlled drugs require prescription authorization."
)

print("\nORIGINAL TEXT:\n")
print(text)

# ---------------------------------------------------
# Failure mode 1: Broken chunks (cut in the middle)
# ---------------------------------------------------
print("\n" + "=" * 70)
print("FAILURE MODE 1: BROKEN CHUNKS")
print("=" * 70)

broken_chunk_1 = "Adult patients should take paracetamol at a dose of 500mg to 1000mg every four to six"
broken_chunk_2 = "hours. Maximum daily dose is 4000mg."

print("\nChunk 1:")
print(broken_chunk_1)

print("\nChunk 2:")
print(broken_chunk_2)

print("\nProblem:")
print("- Chunk 1 ends in the middle of a sentence")
print("- Chunk 2 loses the full context")


# ---------------------------------------------------
# Failure mode 2: Chunk too large (mixed topics)
# ---------------------------------------------------
print("\n" + "=" * 70)
print("FAILURE MODE 2: CHUNK TOO LARGE")
print("=" * 70)

large_chunk = (
    "Paracetamol dosage: 500mg to 1000mg every 4 to 6 hours. Max 4000mg/day. "
    "For wounds, clean with saline and apply dressing. "
    "Controlled drugs require authorization."
)

print("\nLarge Chunk:")
print(large_chunk)

print("\nProblem:")
print("- Multiple unrelated topics are mixed together")
print("- Retrieval for a dosage question may become weak")


# ---------------------------------------------------
# Failure mode 3: Chunk too small (missing context)
# ---------------------------------------------------
print("\n" + "=" * 70)
print("FAILURE MODE 3: CHUNK TOO SMALL")
print("=" * 70)

small_chunk = "Max 4000mg/day."

print("\nSmall Chunk:")
print(small_chunk)

print("\nProblem:")
print("- Looks precise, but context is missing")
print("- Which drug?")
print("- For adults or children?")
print("- Any exceptions?")


# ---------------------------------------------------
# Good chunk example
# ---------------------------------------------------
print("\n" + "=" * 70)
print("GOOD CHUNK EXAMPLE")
print("=" * 70)

good_chunk = (
    "Adult patients should take paracetamol at a dose of 500mg to 1000mg every four to six hours. "
    "Maximum daily dose is 4000mg."
)

print("\nGood Chunk:")
print(good_chunk)

print("\nWhy this is better:")
print("- Sentence is complete")
print("- Topic is focused")
print("- Context is preserved")
