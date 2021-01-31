import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probability = dict()

    for person in people:

        num_of_genes = None #Number of genes considered
        probability[person] = 0.0 #Keep track of every person's computed probability
        probability_mother = 0.0 #Keep track of Mother's probability
        probability_father = 0.0 #Keep track of Father's probability
        mother = people[person]["mother"]
        father = people[person]["father"]
        if person in one_gene:
            #Find the probability that a person has one gene
            num_of_genes = 1
            if mother is None and father is None: #Person does not have any parents, compute based on probs
                probability[person] = PROBS["gene"][num_of_genes]
            else: #Compute the probability based off of parents
                if mother is not None:
                    #Mother can only be in one_gene, two_gene, or neither
                    if mother not in one_gene and mother not in two_genes:
                        probability_mother = PROBS["mutation"] #Mother has zero genes so the chance of passing on the gene is the chance of a mutation
                    elif mother in two_genes:
                        probability_mother = 1.00 - PROBS["mutation"] #Mother has two genes so the chance of passing on the gene is the chance it does not mutate
                    else:
                        #Mother is in one_gene
                        probability_mother = .50 - PROBS["mutation"] + PROBS["mutation"] #50% chance to pass on the one_gene - the prbability + probability that the other does mutate that it does not mutate

                if father is not None:
                    #Father can only be in one_gene, two_gene, or neither
                    if father not in one_gene and father not in two_genes:
                        #Father has no copies of the gene
                        probability_father = PROBS["mutation"]
                    elif father in two_genes:
                        #Father has two copies of the gene
                        probability_father = 1.00 - PROBS["mutation"]
                    else:
                        #Father has one copy of the gene
                        probability_father = .50 - PROBS["mutation"] + PROBS["mutation"]
            #Compute the probability for one gene
                if mother is not None and father is not None: #Compute the probability that the person has the gene based off of parents probability
                    probability[person] = probability_mother * ( 1.00 - probability_father ) + probability_father * ( 1.00 - probability_mother )
                elif mother is None and father is not None:
                    probability[person] = probability_father
                elif mother is not None and father is None:
                    probability[person] = probability_mother

        elif person in two_genes:
            #Find the probability that a person has two genes
            num_of_genes = 2
            if mother is None and father is None: #Person does not have any parents, compute based on probs
                probability[person] = PROBS["gene"][num_of_genes]
            else:
                if mother is not None:
                    if mother not in one_gene and mother not in two_genes:
                        #Mother has no copies of the gene, only way to pass it on is if the gene mutates
                        probability_mother = PROBS["mutation"]
                    elif mother in one_gene and mother not in two_genes:
                        #Mother has one copy, only way to pass it on is if the bad_gene passes on and doesnt mutate or the good gene gets passed on it does mutate
                        probability_mother = .50 - PROBS["mutation"] + PROBS["mutation"]
                    else:
                        #Mother has two copies of the gene
                        probability_mother = 1.00 - PROBS["mutation"]
                if father is not None:
                    if father not in one_gene and father not in two_genes:
                        probability_father = PROBS["mutation"]
                    elif father not in one_gene and father in two_genes:
                        probability_father = 1.00 - PROBS["mutation"]
                    else:
                        probability_father = .50 - PROBS["mutation"] + PROBS["mutation"]

                if mother is not None and father is not None:
                    probability[person] = probability_mother * probability_father
                elif mother is None and father is not None:
                    probability[person] = probability_father
                else:
                    probability[person] = probability_mother

        else:
            #Find the probability that a person has no genes
            num_of_genes = 0
            if mother is None and father is None: #Person does not have any parents, compute based on probs
                probability[person] = PROBS["gene"][num_of_genes]
            else:
                #Compute the probability based off of parents
                if mother is not None:
                    if mother not in one_gene and mother not in two_genes:
                        #Mother has no copies of the gene
                        probability_mother = 1.00 - PROBS["mutation"]
                    elif mother not in one_gene and mother in two_genes:
                        #Mother has two copies of the gene
                        probability_mother = PROBS["mutation"] #Possible if the gene mutates to being inactive
                    else:
                        probability_mother = .50 - PROBS["mutation"] + PROBS["mutation"]
                if father is not None:
                    if father not in one_gene and father not in two_genes:
                        probability_father = 1.00 - PROBS["mutation"]
                    elif father not in one_gene and father in two_genes:
                        probability_father = PROBS["mutation"]
                    else:
                        probability_father = .50 - PROBS["mutation"] + PROBS["mutation"]

                if mother is not None and father is not None:
                    probability[person] = probability_mother * probability_father
                elif mother is None and father is not None:
                    probability[person] = probability_father
                else:
                    probability[person] = probability_mother

        trait = PROBS["trait"][num_of_genes][True] if person in have_trait else PROBS["trait"][num_of_genes][False]
        probability[person] = probability[person] * trait

    joint_prob = 1.0
    for person in probability:
        joint_prob = joint_prob * probability[person]

    return joint_prob

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:

        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:

        normalized_gene = 0.0
        normalized_trait = 0.0
        for i in range(3): #First find the normalizing value
            normalized_gene += probabilities[person]["gene"][i]
        for i in range(3): #Then normalize all the values in the gene distribution
            probabilities[person]["gene"][i] /= normalized_gene

        normalized_trait = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]

        probabilities[person]["trait"][True] /= normalized_gene
        probabilities[person]["trait"][False] /= normalized_gene


if __name__ == "__main__":
    main()
