"""ByTheBookTest: does a model follow the rulebook, or what it already knows?

A protocol to generate test datasets in which a rulebook contradicts world knowledge and every expected
answer is checked with the Z3 theorem prover, and a test program that asks a model every claim through
llama.cpp and measures how far world knowledge interferes with the rulebook.
"""

__version__ = "0.1.0"
