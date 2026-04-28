import os
import random


def sample_smiles(csv_path, n, seed, max_rows=None):
    # If max_rows is provided, sample from the same leading row pool used by VAE
    # training: first max_rows rows, then random.sample from that pool.
    # Otherwise, approximate random sampling across the whole file with byte offsets.
    if n <= 0:
        return []

    rng = random.Random(seed)

    if max_rows is not None:
        pool = []
        seen = set()
        with open(csv_path, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                smi = line.strip()
                if smi and smi not in seen:
                    seen.add(smi)
                    pool.append(smi)
                if i + 1 >= max_rows:
                    break

        if len(pool) <= n:
            rng.shuffle(pool)
            return pool
        return rng.sample(pool, n)

    file_size = os.path.getsize(csv_path)
    samples = []
    seen = set()

    with open(csv_path, "rb") as f:
        f.readline()
        data_start = f.tell()
        if data_start >= file_size:
            return []

        max_attempts = max(1_000, n * 50)
        attempts = 0
        while len(samples) < n and attempts < max_attempts:
            attempts += 1
            f.seek(rng.randrange(data_start, file_size))
            f.readline()  # discard partial line
            line = f.readline()
            if not line:
                f.seek(data_start)
                line = f.readline()

            smi = line.decode("utf-8", errors="ignore").strip()
            if smi and smi not in seen:
                seen.add(smi)
                samples.append(smi)

        if len(samples) < n:
            f.seek(data_start)
            for line in f:
                smi = line.decode("utf-8", errors="ignore").strip()
                if smi and smi not in seen:
                    seen.add(smi)
                    samples.append(smi)
                if len(samples) >= n:
                    break

    rng.shuffle(samples)
    return samples
