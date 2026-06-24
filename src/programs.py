"""
Program bank: 8 CruxEval-style output-prediction programs.
Each is small, self-contained, and semantically verifiable by inspection.

Rationale for output-prediction task (from MUCOCO Table 5):
output prediction has 18.99% inconsistency rate -- high enough to be
meaningful, and evaluable with simple string match (no test suite execution).
"""

PROGRAMS = [
    {
        "id": "P1",
        "desc": "Return True if n is even, False otherwise.",
        "code": (
            "def check_even(n):\n"
            "    return n % 2 == 0"
        ),
        "inp": "4",
        "exp": "True",
    },
    {
        "id": "P2",
        "desc": "Return the sum of a list of numbers.",
        "code": (
            "def sum_list(nums):\n"
            "    total = 0\n"
            "    for x in nums:\n"
            "        total += x\n"
            "    return total"
        ),
        "inp": "[1, 2, 3, 4, 5]",
        "exp": "15",
    },
    {
        "id": "P3",
        "desc": "Count the number of vowels in a string.",
        "code": (
            "def count_vowels(s):\n"
            "    count = 0\n"
            "    for c in s:\n"
            "        if c in 'aeiouAEIOU':\n"
            "            count += 1\n"
            "    return count"
        ),
        "inp": "'hello world'",
        "exp": "3",
    },
    {
        "id": "P4",
        "desc": "Reverse a string.",
        "code": (
            "def reverse_string(s):\n"
            "    return s[::-1]"
        ),
        "inp": "'abcde'",
        "exp": "'edcba'",
    },
    {
        "id": "P5",
        "desc": "Return the maximum value in a list.",
        "code": (
            "def find_max(lst):\n"
            "    result = lst[0]\n"
            "    for item in lst:\n"
            "        if item > result:\n"
            "            result = item\n"
            "    return result"
        ),
        "inp": "[3, 1, 4, 1, 5, 9, 2, 6]",
        "exp": "9",
    },
    {
        "id": "P6",
        "desc": "Check if a string is a palindrome.",
        "code": (
            "def is_palindrome(s):\n"
            "    return s == s[::-1]"
        ),
        "inp": "'racecar'",
        "exp": "True",
    },
    {
        "id": "P7",
        "desc": "Return the factorial of n.",
        "code": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(1, n + 1):\n"
            "        result *= i\n"
            "    return result"
        ),
        "inp": "5",
        "exp": "120",
    },
    {
        "id": "P8",
        "desc": "Return a list with duplicate elements removed, preserving order.",
        "code": (
            "def remove_duplicates(lst):\n"
            "    seen = []\n"
            "    for item in lst:\n"
            "        if item not in seen:\n"
            "            seen.append(item)\n"
            "    return seen"
        ),
        "inp": "[1, 2, 2, 3, 3, 3, 4]",
        "exp": "[1, 2, 3, 4]",
    },
]
