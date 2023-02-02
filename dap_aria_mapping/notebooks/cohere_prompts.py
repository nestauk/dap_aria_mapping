# %%
import json
import numpy as np
from dap_aria_mapping.getters.taxonomies import get_topic_names
from dap_aria_mapping import PROJECT_DIR
import cohere
from collections import defaultdict
from tqdm import tqdm
from time import sleep

co = cohere.Client("pArhjogTnlZGIUEocMpv1lVZKv3RHsf7KIgTPk0F")

prompt = f"""
This program generates a topic label given a list of entities contained within a topic.

Topic entities: Probability space, Sample space, Event, Elementary event, Mutual exclusivity, Outcome, Singleton, Experiment, Bernoulli trial, Probability distribution, Bernoulli distribution, Binomial distribution, Normal distribution, Probability measure.
Topic label: Probability theory.

--
Topic entities: Chromosome, DNA, RNA, Genome, Heredity, Mutation, Nucleotide, Variation.
Topic label: Genetics.

--
Topic entities: Breakbeat, Chiptune, Dancehall, Downtempo, Drum and bass, Dub, Dubstep, Electro, EDM, Grime, Hardcore, House, IDM, Reggaeton, Synth-pop, Techno.
Topic label: Electronic music.

--
Topic entities: Ruy Lopez, French Defence, Petrov's Defence, Vienna Game, Centre Game, King's Gambit, Philidor Defence, Giuoco Piano, Evans Gambit, Hungarian Defence, Scotch Game, Four Knights Game, King's Pawn Opening.
Topic label: Chess openings.

--
Topic entities: Arial, Verdana, Tahoma, Trebuchet, Times New Roman, Georgia, Garamond, Courier New.
Topic label: Typefaces.

--
Topic entities: Amharic, Dinka, Ibo, Kirundi, Mandinka, Nuer, Oromo, Swahili, Tigrigna, Wolof, Xhosa, Yoruba, Zulu.
Topic label: African languages.

--
Topic entities: Algorithm, Mathematical optimization, Machine learning, Classical mechanics, Geometry.
Topic label: Mathematics and Computer Science.

--
Topic entities: Monet, Renoir, Degas, Cezanne, Manet, Toulouse-Lautrec, Van Gogh, Gauguin, Pissarro, Sisley.
Topic label: Impressionist artists.

--
Topic entities: Pythagoras, Euclid, Archimedes, Apollonius, Plato, Aristotle, Hippocrates, Galen, Ptolemy.
Topic label: Ancient Greek mathematicians and philosophers.

--
Topic entities: Amazon, Google, Apple, Facebook, Microsoft, Alibaba, Tencent, Tesla, Netflix, Oracle.
Topic label: Technology companies.
--

Topic entities: Lagrange, Hamilton, Poisson, Cauchy, Gauss, Riemann, Noether, Euler, Leibniz, Newton.
Topic label: 18th and 19th century mathematicians and physicists.
--
Topic entities: Beethoven, Mozart, Chopin, Bach, Tchaikovsky, Haydn, Brahms, Schubert, Handel, Wagner.
Topic label: Classical composers.

--
Topic entities: Fossils, Extinction, Adaptation, Natural selection, Evolution, Paleontology, Taxonomy, Darwin, Mendel.
Topic label: Evolutionary biology.

--
Topic entities: Plate tectonics, Earthquake, Volcano, Tsunami, Magma, Lava, Geology, Seismology, Mineralogy.
Topic label: Earth sciences.

--
Topic entities: Keynes, Marx, Friedman, Smith, Hayek, Schumpeter, Malthus, Ricardo, Hegel, Adam Smith.
Topic label: Economists and economic theories.

--
Topic entities: Relativity, Quantum mechanics, Electromagnetism, Thermodynamics, Astrophysics, Cosmology, Particle physics, String theory.
Topic label: Physics.

--
Topic entities: Shakespeare, Tolstoy, Dante, Chaucer, Austen, Hemingway, Whitman, Faulkner, Orwell, Camus.
Topic label: Classic authors.

--
Topic entities: Mona Lisa, Sistine Chapel, The Last Supper, The Starry Night, The Persistence of Memory, The Scream, The Kiss, The Dance, The Water Lilies.
Topic label:Famous paintings and artists.

--

"""


def evaluate_entities(dictionary, key, val):
    eval = prompt + "Topic_entities: " + val[3:] + "\n\nTopic_label:"

    response = co.generate(
        model="xlarge",
        prompt=eval,
        max_tokens=20,
        num_generations=5,
        temperature=0.6,
        stop_sequences=["--"],
    )
    topic_label = response.generations[0].text
    label_processed = (topic_label.split("\n")[0]).strip()
    dictionary.update({f"Topic {key}": label_processed})


taxonomy_class = "entity"

for level in range(1, 6):
    dict_names = defaultdict(list)
    dict_entities = get_topic_names("cooccur", taxonomy_class, level, True)

    for key, val in tqdm(dict_entities.items()):
        sleep(np.random.uniform(1, 5))
        try:
            evaluate_entities(dictionary=dict_names, key=key, val=val)
        except:
            dict_names.update({"Topic " + key: "Error"})
    with open(
        PROJECT_DIR
        / "outputs"
        / f"cohere_labels_class_{taxonomy_class}_level_{level}.json",
        "w",
    ) as f:
        json.dump(dict_names, f)
# %%
