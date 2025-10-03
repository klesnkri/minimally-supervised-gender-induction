# Minimally Supervised Induction of Grammatical Gender

## What the project is solving
I have tried to reimplement the paper [Minimally Supervised Induction of Grammatical Gender](https://aclanthology.org/N03-1006.pdf) and test it on the Czech Prague Dependency Treebank corpus.

## What was implemented
### Language data
To simplify the process of obtaining language data, I only used the Czech Prague Dependency Treebank corpus.
Instead of relying on a dictionary, I extracted nouns in their base forms along with their reference genders directly from this corpus.
I then used the same corpus, ignoring the annotations, as the text corpus for running the algorithm.

### Seeding
I did not test the method of translingual projection of natural gender for seeding, as Czech includes masculine inanimate and neuter genders.
Instead, I employed the approach that automatically selects seeds based on frequency, the number of contexts in which they co-occur, and suffix patterns.
However, I’m not entirely clear on how selecting nouns based on suffix patterns works, so I did not implement that aspect.

### Context bootstrapping
I have tried to implement the context bootstrapping algorithm as described in the paper.
When filtering contexts, I haven't performed filtering using a frequency threshold sensitive to both the size of the corpus and the seed list, since I don't think it's needed.

### Morphological analysis
I have tried to implement the morphological analysis algorithm as described in the paper.

### Treatment of unknown words
As the preceding steps of the algorithm cover almost all the nouns, it doesn't influence the results much, which gender is chosen as the default gender.

## How to run the project
1. Create Python virtual environment: `python3.11 -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate`
3. Install requirements: `pip3 install -r requirements.txt`
4. Run the project: `make run`

## Sample output
Outputs of the algorithm with various hyperparameters can be found in the `out` directory.
On every run of the algorithm, new logging directory is created.
The logging files contain the final gender assignment, used hyperparameters and statistics (coverage and accuracy) after every step of the gender induction algorithm.
