import asyncio
import aiohttp
import time
import random
import sys
import csv

# --- Configuration ---
API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "Qwen/Qwen3-0.6B"
CONCURRENT_REQUESTS = 1346
MAX_TOKENS = 100
TOTAL_PROMPTS = 2000
OUTPUT_FILE = "output.csv"

# --- Word sets ---
set1 = [
    "calm","bright","golden","silent","gentle","vivid","tender","soft","mellow","serene",
    "warm","brisk","steady","lucid","bold","amber","quiet","fresh","nimble","radiant",
    "crisp","lunar","rustic","tranquil","pure","subtle","gracious","merry","silver","lively",
    "poised","blue","humble","sweet","hardy","noble","glossy","misty","sunny","leafy",
    "brave","playful","clever","soothing","shining","peaceful","polished","velvet","airy","clear"
]

set2 = [
    "river","garden","forest","valley","harbor","meadow","lantern","cloud","mirror","circle",
    "sunrise","orchard","horizon","stream","starlight","path","breeze","mountain","island","glimmer",
    "field","harvest","coral","petal","echo","comet","canvas","ridge","brook","ember",
    "rainbow","moonlight","stone","wave","feather","dawn","willow","signal","rain","spark",
    "trail","waterfall","mist","bloom","glade","shore","village","treasure","meadowlark","sunbeam"
]

set3 = [
    "carries","paints","guides","reveals","holds","shapes","whispers","awakens","frames","reflects",
    "lifts","spreads","softens","balances","shimmers","welcomes","builds","stirs","gathers","reminds",
    "heals","follows","protects","blooms","drifts","unfolds","glows","rises","settles","wanders",
    "listens","sings","breathes","glides","grows","threads","sparkles","flows","rests","returns",
    "opens","touches","nourishes","kindles","shelters","connects","echoes","surrounds","merges","lasts"
]

set4 = [
    "light","dreams","silence","joy","warmth","peace","color","music","memory","hope",
    "hearts","stories","seasons","breath","glory","beauty","meaning","harmony","wonder","time",
    "comfort","grace","motion","laughter","promise","morning","evening","pages","waves","flowers",
    "notes","truth","wildflowers","glances","paths","echoes","shadows","lanterns","footsteps","tomorrow",
    "moonbeams","tides","petals","gold","silver","fragrance","whispers","sunlight","kindness","serenity"
]

def generate_prompt():
    return f"{random.choice(set1)} {random.choice(set2)} {random.choice(set3)} {random.choice(set4)}"

# --- Pre-generate prompts ---
random.seed(42)
PROMPT_POOL = [generate_prompt() for _ in range(TOTAL_PROMPTS)]

def get_prompt(request_id):
    return PROMPT_POOL[request_id % TOTAL_PROMPTS]

async def send_request(session, request_id):
    prompt = get_prompt(request_id)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.8,
        "ignore_eos": True
    }

    start_time = time.time()
    try:
        async with session.post(API_URL, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                generated_tokens = result.get("usage", {}).get("completion_tokens", 0)
                output_text = result.get("choices", [{}])[0].get("text", "")
                elapsed_time = time.time() - start_time
                return True, prompt, output_text, generated_tokens, elapsed_time
            else:
                print(f"[{request_id}] Status: {response.status}")
                return False, prompt, "", 0, 0
    except Exception as e:
        print(f"[{request_id}] Failed: {e}")
        return False, prompt, "", 0, 0

async def main():
    print(f"🚀 Starting load test with {CONCURRENT_REQUESTS} requests")

    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [send_request(session, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # --- Metrics ---
    successful = [r for r in results if r[0]]
    successful_requests = len(successful)
    total_tokens = sum(r[3] for r in successful)

    if successful_requests > 0:
        print("📊 Results")
        print(f"Time: {total_time:.2f}s   ")
        print(f"Success: {successful_requests}/{CONCURRENT_REQUESTS}  ")
        print(f"Throughput: {successful_requests/total_time:.2f} req/s  ")
        print(f"Token Throughput: {total_tokens/total_time:.2f} tok/s  ")

    # --- Save to CSV ---
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "output", "tokens", "latency"])

        for success, prompt, output, tokens, latency in results:
            if success:
                writer.writerow([prompt, output.strip(), tokens, latency])

    print(f"💾 Saved results to {OUTPUT_FILE}  ")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())