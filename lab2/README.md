# Strategy

1. Remove duplicate lists from the ones produced by the problem function
2. Build a population containing individuals composed of 1 random tuple+the tuples that will be certainly part of the solution (when a tuple is the only one to cover a certain number, that tuple will certainly be part of the solution)
3. In each generation create new offsprings by either:
    1. Choosing 1 tuple by tournament selection and mutating it
    2. Choosing 2 tuples by tournament selection and recombining them
4. Add the resulting offsprings to the population and discard the less fit individuals
5. Save best individual found so far
6. The evolution continues until a max number of generation is reached, or we reach a steady state (the best individual hasn't changed for many generations)

# Mutation

- remove 0 or more tuples from the genome (only if it already covers all numbers)
- add 1 or more tuples not already present in the genome

# Recombination

- remove 0 or more tuples from the longest parent (only if it already covers all numbers)
- add one of the non-common tuples of the shorter parent to the longest parent

# Performances

| N    | Solution length | Time taken (s) |
| ---- | --------------- | -------------- |
| 5    | 5               | 0.875          |
| 10   | 10              | 0.586          |
| 20   | 24              | 2.049          |
| 30   | 40              | 8.768          |
| 40   | 62              | 7.626          |
| 50   | 78              | 15.738         |
| 100  | 176             | 48.195         |
| 500  | 1449            | 290.374        |
| 1000 | 3036            | 887.792        |
