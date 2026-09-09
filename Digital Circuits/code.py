"""
This program takes:
    - number of variables
    - the list of MINTERMS (positions where F = 1)
    - the list of DON'T CARE terms (positions where F can be 0 or 1)
and produces the most optimal (minimal) SUM-OF-PRODUCTS expression.

Example:
    F = Sum of m(0, 2, 3, 6, 7) + d(4, 5)

No external "logic gate" or boolean-algebra library is used.
Everything (binary conversion, term combination, prime implicant
chart, essential PI selection) is done manually with plain
Python data structures (strings, sets, dictionaries) so the
logic is fully visible and easy to follow.
=====================================================================
"""

from itertools import combinations


# ---------------------------------------------------------------
# STEP 1: Convert a decimal number into an n-bit binary string
# ---------------------------------------------------------------
def to_binary(number, num_vars):
    # format() pads the binary representation with leading zeros
    # so every term has the same length (= number of variables)
    return format(number, '0{}b'.format(num_vars))


# ---------------------------------------------------------------
# STEP 2: Check if two binary strings (terms) differ in EXACTLY
#          one bit position. If yes, return that position.
#          If they differ in 0 or 2+ positions, they cannot combine.
#          ('-' means "don't care bit", already merged earlier and
#           must line up in the same position in both terms)
# ---------------------------------------------------------------
def differing_position(term1, term2):
    diff_count = 0
    diff_index = -1
    for i in range(len(term1)):
        if term1[i] != term2[i]:
            diff_count += 1
            diff_index = i
            if diff_count > 1:
                return -1          # more than 1 difference -> cannot combine
    return diff_index if diff_count == 1 else -1


# ---------------------------------------------------------------
# STEP 3: Combine two terms that differ in one bit by placing a
#          '-' (dash) in that differing position.
#          e.g. 011 and 111 differ at position 0 -> combined = -11
# ---------------------------------------------------------------
def combine_terms(term1, position):
    return term1[:position] + '-' + term1[position + 1:]


# ---------------------------------------------------------------
# STEP 4: Core Quine-McCluskey reduction.
#          Repeatedly combine terms that differ by 1 bit.
#          Any term that never gets combined with another term
#          becomes a PRIME IMPLICANT (cannot be simplified further).
# ---------------------------------------------------------------
def find_prime_implicants(all_terms, num_vars):
    # Each "term" is stored as a tuple: (binary_string, set_of_source_minterms)
    current_terms = [(to_binary(t, num_vars), frozenset([t])) for t in all_terms]

    prime_implicants = set()   # will hold (binary_string, frozenset(sources))

    while True:
        combined_this_round = set()   # terms that got merged this round
        next_terms = {}                # dict avoids storing duplicate binary strings

        # Try every pair of terms once
        for (b1, s1), (b2, s2) in combinations(current_terms, 2):
            pos = differing_position(b1, b2)
            if pos != -1:
                new_binary = combine_terms(b1, pos)
                new_sources = s1 | s2
                next_terms[new_binary] = new_sources
                combined_this_round.add((b1, s1))
                combined_this_round.add((b2, s2))

        # Any term NOT combined this round is final -> a prime implicant
        for term in current_terms:
            if term not in combined_this_round:
                prime_implicants.add(term)

        if not next_terms:
            break  # no more combining possible, we are done

        current_terms = list(next_terms.items())

    return prime_implicants


# ---------------------------------------------------------------
# STEP 5: Does a prime implicant (with dashes) COVER a given
#          minterm's binary representation?
#          Every non-dash bit must match exactly.
# ---------------------------------------------------------------
def covers(pattern, minterm_binary):
    for p_bit, m_bit in zip(pattern, minterm_binary):
        if p_bit != '-' and p_bit != m_bit:
            return False
    return True


# ---------------------------------------------------------------
# HELPER: Count how many actual variables (literals) appear in a
#          pattern, i.e. how many bits are NOT a dash '-'.
#          Fewer literals = simpler term = fewer variables used.
#          e.g. "1-0" has 2 literals, "---" has 0 literals.
# ---------------------------------------------------------------
def count_literals(pattern):
    return sum(1 for bit in pattern if bit != '-')


# ---------------------------------------------------------------
# HELPER: Remove any set from a collection of sets that is a
#          SUPERSET of another set in the same collection.
#          (This is the Absorption Law: if set A already implies
#          set B, keeping the bigger A is redundant.)
#          Keeps Petrick's method from blowing up in size.
# ---------------------------------------------------------------
def remove_redundant_supersets(list_of_sets):
    result = []
    for s in list_of_sets:
        if not any(other < s for other in list_of_sets):
            result.append(s)
    # de-duplicate identical sets
    unique = []
    for s in result:
        if s not in unique:
            unique.append(s)
    return unique


# ---------------------------------------------------------------
# STEP 6b: PETRICK'S METHOD
#          Finds the truly OPTIMAL way to cover the remaining
#          (non-essential) minterms -- not just a greedy guess.
#          It works by building a boolean product-of-sums of PI
#          choices, one "sum" (clause) per uncovered minterm, then
#          multiplying all clauses out into every possible valid
#          covering combination, and finally picking the smallest
#          one: first by NUMBER OF TERMS, then (to also minimize
#          the number of variables) by TOTAL LITERAL COUNT.
# ---------------------------------------------------------------
def petricks_method(remaining_minterms, chart, pi_list):
    # One clause per uncovered minterm: the set of PI indexes that
    # can cover it. e.g. minterm 5 might be coverable by PI #2 or #4
    clauses = [frozenset(chart[m]) for m in remaining_minterms]

    # Start by "expanding" the first clause: each single choice is
    # itself a valid (partial) covering combination so far.
    possible_covers = [frozenset([idx]) for idx in clauses[0]]

    # Multiply in every remaining clause (like expanding
    # (a+b)(c+d)(e+f)... into a sum of product terms)
    for clause in clauses[1:]:
        expanded = set()
        for existing_combo in possible_covers:
            for idx in clause:
                expanded.add(existing_combo | frozenset([idx]))
        possible_covers = remove_redundant_supersets(list(expanded))

    # Among ALL valid covering combinations, pick the best one:
    #   1st priority -> fewest prime implicants (fewest terms)
    #   2nd priority -> fewest total literals (fewest variables)
    best_combo = min(
        possible_covers,
        key=lambda combo: (
            len(combo),
            sum(count_literals(pi_list[i][0]) for i in combo)
        )
    )
    return best_combo


# ---------------------------------------------------------------
# STEP 6: Build the Prime Implicant Chart and select the
#          minimum set of prime implicants that covers every
#          required minterm.
#          - Essential PIs are picked first (forced choices).
#          - Petrick's Method is used for whatever minterms are
#            left, guaranteeing the fewest terms AND the fewest
#            total variables (literals) in the final expression.
# ---------------------------------------------------------------
def select_minimal_cover(prime_implicants, minterms, num_vars):
    pi_list = list(prime_implicants)
    minterm_bins = {m: to_binary(m, num_vars) for m in minterms}

    # chart[minterm] = list of prime implicant indexes covering it
    chart = {m: [] for m in minterms}
    for idx, (pattern, _) in enumerate(pi_list):
        for m in minterms:
            if covers(pattern, minterm_bins[m]):
                chart[m].append(idx)

    chosen = set()
    covered_minterms = set()

    # --- 6a: Essential Prime Implicants ---
    # A minterm that is covered by only ONE prime implicant means
    # that PI is "essential" -- it MUST be part of the final answer.
    for m, pi_indexes in chart.items():
        if len(pi_indexes) == 1:
            chosen.add(pi_indexes[0])

    for idx in chosen:
        covered_minterms |= pi_list[idx][1] & set(minterms)

    # --- 6b: Petrick's Method for whatever remains ---
    remaining = set(minterms) - covered_minterms
    if remaining:
        chosen |= petricks_method(remaining, chart, pi_list)

    return [pi_list[i][0] for i in chosen]


# ---------------------------------------------------------------
# STEP 7: Convert a binary pattern (with dashes) into a readable
#          product term, e.g. "1-0" with vars A,B,C -> "A C'"
# ---------------------------------------------------------------
def pattern_to_expression(pattern, num_vars):
    var_names = [chr(ord('A') + i) for i in range(num_vars)]
    literals = []
    for bit, var in zip(pattern, var_names):
        if bit == '1':
            literals.append(var)
        elif bit == '0':
            literals.append(var + "'")
        # '-' means this variable does not appear in the term
    return ''.join(literals) if literals else '1'


# ---------------------------------------------------------------
# STEP 7b: Convert a binary pattern (with dashes) into a readable
#          SUM term for POS form, e.g. "1-0" with vars A,B,C
#          -> "(A' + C)"
#          NOTE: for POS the bit sense is FLIPPED compared to SOP
#          because we are simplifying the ZEROS of the function
#          (maxterms). A '1' bit means the variable is complemented
#          in the sum term, a '0' bit means the plain variable.
# ---------------------------------------------------------------
def pattern_to_sum_term(pattern, num_vars):
    var_names = [chr(ord('A') + i) for i in range(num_vars)]
    literals = []
    for bit, var in zip(pattern, var_names):
        if bit == '0':
            literals.append(var)
        elif bit == '1':
            literals.append(var + "'")
        # '-' means this variable does not appear in this sum term
    return '(' + ' + '.join(literals) + ')' if literals else '(1)'


# ---------------------------------------------------------------
# STEP 8: Full pipeline -> given minterms + don't cares, return
#          the final minimized Sum-of-Products (SOP) expression.
#          This is the "minimizer": F = m1 + m2 + m3 ...
# ---------------------------------------------------------------
def minimize(num_vars, minterms, dont_cares=None):
    dont_cares = dont_cares or []
    all_terms = list(minterms) + list(dont_cares)

    prime_implicants = find_prime_implicants(all_terms, num_vars)

    # Only the ORIGINAL minterms (not don't cares) must be covered.
    final_patterns = select_minimal_cover(prime_implicants, minterms, num_vars)

    expressions = [pattern_to_expression(p, num_vars) for p in final_patterns]
    return ' + '.join(expressions)


# ---------------------------------------------------------------
# STEP 9: Full pipeline -> given MAXTERMS + don't cares, return
#          the final minimized Product-of-Sums (POS) expression.
#          This is the "maximizer": F = M1 . M2 . M3 ...
#
#          Idea: POS minimization uses the exact same
#          Quine-McCluskey machinery (find_prime_implicants /
#          select_minimal_cover) but applied to the places where
#          F = 0 (the maxterms) instead of where F = 1. Each
#          resulting binary pattern is then written as a SUM term
#          (with the bit sense flipped) instead of a product term,
#          and the sum terms are combined with AND instead of OR.
# ---------------------------------------------------------------
def maximize(num_vars, maxterms, dont_cares=None):
    dont_cares = dont_cares or []
    all_terms = list(maxterms) + list(dont_cares)

    # Same prime-implicant search, just run on the maxterms (0-positions)
    prime_implicants = find_prime_implicants(all_terms, num_vars)

    # Same essential + greedy cover selection, just covering maxterms
    final_patterns = select_minimal_cover(prime_implicants, maxterms, num_vars)

    sum_terms = [pattern_to_sum_term(p, num_vars) for p in final_patterns]
    return ' . '.join(sum_terms)


# ---------------------------------------------------------------
# MAIN: simple interactive input, OR run the example from the
#        board: F = Sum of m(0,2,3,6,7) + d(4,5)
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Quine-McCluskey Boolean Expression Minimizer / Maximizer")
    print("----------------------------------------------------------")
    mode = input("Choose mode -> (1) Minimize SOP from minterms  "
                  "(2) Maximize POS from maxterms: ").strip()

    n = int(input("Enter number of variables: "))

    if mode == '2':
        # POS mode: user supplies MAXTERMS (positions where F = 0)
        terms = list(map(int, input("Enter maxterms (space separated): ").split()))
        dc_input = input("Enter don't care terms (space separated, or leave blank): ").strip()
        dcs = list(map(int, dc_input.split())) if dc_input else []

        result = maximize(n, terms, dcs)
        print("\nMinimized POS Expression:")
        print("F =", result)
    else:
        # SOP mode: user supplies MINTERMS (positions where F = 1)
        terms = list(map(int, input("Enter minterms (space separated): ").split()))
        dc_input = input("Enter don't care terms (space separated, or leave blank): ").strip()
        dcs = list(map(int, dc_input.split())) if dc_input else []

        result = minimize(n, terms, dcs)
        print("\nMinimized SOP Expression:")
        print("F =", result)
