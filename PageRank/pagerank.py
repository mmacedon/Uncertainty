import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ####Delete Line below when finished
    print(corpus)


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict() #Create a dictionary for the probability distribution
    all_pages = []
    links = list(corpus[page]) #All pages linked by current page
    for u in corpus:
        all_pages.append(u)

    #If the current page does not link to any other pages, all pages are equally likely
    if len(links) == 0:
        print("Current Page does not have any links")
        probability = 1.0 / len(all_pages)
        for current_page in all_pages:
            distribution[current_page] = probability
        return distribution

    damping_probability = damping_factor / len(links) #Probability that a linked page will be selected
    nondamping_probability = (1.0 - damping_factor) / len(all_pages) #Probability that a page will be randomly selected

    for current_page in all_pages:
        distribution[current_page] = nondamping_probability

    for current_page in all_pages:
        if current_page in links:
            distribution[current_page] = distribution[current_page] + damping_probability

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = []
    first_sample_prob = random.randint(0, len(corpus) - 1)
    distribution_count = dict()

    for u in corpus:
        distribution_count[u] = 0
        all_pages.append(u)

    sample = all_pages[first_sample_prob]
    for i in range(n - 1): # n - 1 because first sample was already calculated
        selection_bucket = dict()
        selection_start = 0.0
        sample_distribution = transition_model(corpus, sample, damping_factor)
        sample_prob = random.random()
        for u in sample_distribution:
            floor = selection_start
            ceiling = selection_start + sample_distribution[u]
            selection_start = ceiling
            selection_bucket[u] = [floor, ceiling]
        for u in selection_bucket:
            v = selection_bucket[u]
            if v[0] < sample_prob < v[1]:
                sample = u
                distribution_count[u] += 1
    distribution = dict()
    for u in distribution_count:
        distribution[u] = float(distribution_count[u]) / n

    return distribution

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    distribution = dict()
    corpus_length = len(corpus)
    for u in corpus: #On first iteration, each page is equally likely.
        distribution[u] = 1.0 / corpus_length

    difference = 1.0
    max_difference = 0.0
    while ( difference > 0.001 ):
        old_distribution = distribution.copy()
        for u in corpus: #Page we are currently looking at
            prob = (1.0 - damping_factor) / corpus_length
            for x in corpus:
                if u == x:
                    continue
                if u in corpus[x]:
                    links = list(corpus[x])
                    prob += damping_factor * (distribution[x] / len(links))
            distribution[u] = prob
            difference = abs(distribution[u] - old_distribution[u])
            if difference > max_difference: max_difference = difference
    return distribution

if __name__ == "__main__":
    main()
