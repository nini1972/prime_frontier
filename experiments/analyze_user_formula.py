import sys
import os
import io

# Set stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llm_client import query_math_model

prompt = '''A number theory researcher has proposed an empirical method for counting primes in a range [0, N], starting from N = 100 and working bottom-up:

User Observation for N = 100:
1. Base primes are all primes <= sqrt(100) = 10. Odd base primes are {3, 5, 7}. (2 is the even prime).
2. Consider odd prime powers p^k <= 100 (k >= 2):
   - 7 has highest power 7^2 <= 100 (k=2) -> contributes 7
   - 5 has highest power 5^2 <= 100 (k=2) -> contributes 5
   - 3 has highest power 3^4 <= 100 (k=4) -> contributes 3 + 3 + 3 = 9 (three 3s, for powers 3^2, 3^3, 3^4)
3. Summing these values:
   7 + 5 + 3 + 3 + 3 = 21.
4. Adding the 3 base primes: 21 + 3 = 24 (the exact number of odd primes <= 100).
5. Adding the even prime 2: 24 + 1 = 25 = pi(100) (the exact total number of primes <= 100!).

The researcher asks:
'The difficulty is to know which odd numbers are primes and which are not. That is an exercise for the models. If you start with a small range, it might be easier to find... Can we find the mathematical formula for this?'

Please provide a deep, rigorous mathematical analysis:
1. Deconstruct the user's calculation: why did 7 + 5 + 3 + 3 + 3 + 3 + 1 produce exactly 25 for N=100? Is this an exact sieve identity, an approximation, or a coincidence of small numbers?
2. Test this rule on smaller ranges: N = 9, 16, 25, 36, 49, 64, 81. Show what happens.
3. Connect this intuition to classical Sieve Theory: Legendre's formula, Meissel-Lehmer algorithm, and counting odd composites via the Principle of Inclusion-Exclusion.
4. Provide a formalized mathematical formula that accurately realizes the user's bottom-up vision (distinguishing odd primes from odd composites using base primes <= sqrt(N)).'''

print("Sending prompt to Qwen 2.5 Math / R1...")
res = query_math_model(prompt, model_alias="qwen_math", max_tokens=3000)
output_path = os.path.join(os.path.dirname(__file__), "model_analysis.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"MODEL: {res.get('model')}\n")
    f.write(f"SUCCESS: {res.get('success')}\n")
    f.write("=== CONTENT ===\n")
    f.write(str(res.get("content") or ""))
    f.write("\n=== REASONING ===\n")
    f.write(str(res.get("reasoning") or ""))

print(f"Results written to {output_path}")
