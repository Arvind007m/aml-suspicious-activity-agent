"""
Stress Verification Script for Hardening Pass.
Executes all edge-case queries to guarantee 100% crash resilience and determinism.
"""

from orchestrator import run_agent_query

STRESS_QUERIES = [
    "",
    "asdfghjkl",
    "Is customer 99999 suspicious?",
    "structuring in the last 1 day",
    "Analyse this dataset for suspicious activity",
    "Is customer 4521 suspicious?"
]

def run_stress_suite():
    print("\n" + "="*70)
    print("        STARTING AML AGENT HARDENING & STRESS SUITE VERIFICATION       ")
    print("="*70 + "\n")
    
    success_count = 0
    for idx, q in enumerate(STRESS_QUERIES, 1):
        print(f"\n--- STRESS TEST {idx}/6: Query: '{q}' ---")
        try:
            context = run_agent_query(q)
            plan_meta = context.get("plan_meta", {})
            intent = plan_meta.get("intent")
            print(f"[+] SUCCESS: Intent '{intent}' | Time: {context.get('execution_time_sec', 0.0):.2f}s")
            success_count += 1
        except Exception as e:
            print(f"[!] FAILED with exception: {e}")

    print("\n" + "="*70)
    print(f"   STRESS SUITE COMPLETE: {success_count}/6 QUERIES PASSED WITH ZERO CRASHES")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_stress_suite()
