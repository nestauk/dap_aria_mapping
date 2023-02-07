## Outputs from prompts

This short markdown file reports results from prompt engineering explorations using cohere's [API](https://docs.cohere.ai/reference/generate) to label topics from any taxonomy given a set of most frequent topic words. An initial prompt is presented with a series of entities that could co-occur in a topic, and the `generate `endpoint of cohere's API is used to complete the label for the last set of entities.

In order to compare its performance, cohere's API is compared with state-of-the-art generator models Github Copilot and OpenAI's chatGPT. The resulting labels are reported for the initial prompt as well as nine additional prompts, which combine the original prompt as well as any prior additional topic groups.

##### Prompt 1

```
prompt = f"""This program generates a topic label given a list of entities contained within a topic.

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
Topic label:"""
```

Outcomes:

- **chatGPT:** Mathematics and Computer Science.
- **Github Copilot**: Mathematics.
- **cohere:** Computer vision.

##### Prompt 2

```
prompt += f"""Topic entities: Monet, Renoir, Degas, Cezanne, Manet, Toulouse-Lautrec, Van Gogh, Gauguin, Pissarro, Sisley.
Topic label: Impressionist artists.

--
Topic entities: Pythagoras, Euclid, Archimedes, Apollonius, Plato, Aristotle, Hippocrates, Galen, Ptolemy.
Topic label: Ancient Greek mathematicians and philosophers.

--
Topic entities: Amazon, Google, Apple, Facebook, Microsoft, Alibaba, Tencent, Tesla, Netflix, Oracle.
Topic label:"""
```

Outcomes:

- **chatGPT:** Tech companies.
- **Github Copilot**: Technology companies.
- **cohere:** Large tech companies.

##### Prompt 3

```
prompt += f"""Topic entities: Lagrange, Hamilton, Poisson, Cauchy, Gauss, Riemann, Noether, Euler, Leibniz, Newton.
Topic label:"""

```

Outcomes:

- **chatGPT:** Mathematicians of classical mechanics and calculus.
- **Github Copilot**: 18th and 19th century mathematicians and physicists..
- **cohere:** Mathematicians.

##### Prompt 4

```
prompt += f"""Topic entities: Beethoven, Mozart, Chopin, Bach, Tchaikovsky, Haydn, Brahms, Schubert, Handel, Wagner.
Topic label:"""

```

Outcomes:

- **chatGPT:** Classical composers.
- **Github Copilot**: Classical composers.
- **cohere:** Composers.

##### Prompt 5

```
prompt += f"""Topic entities: Fossils, Extinction, Adaptation, Natural selection, Evolution, Paleontology, Taxonomy, Darwin, Mendel.
Topic label:"""

```

Outcomes:

- **chatGPT:** Evolutionary biology.
- **Github Copilot**: Evolutionary biology.
- **cohere:** Evolutionary biology.

##### Prompt 6

```
prompt += f"""Topic entities: Plate tectonics, Earthquake, Volcano, Tsunami, Magma, Lava, Geology, Seismology, Mineralogy.
Topic label:"""

```

Outcomes:

- **chatGPT:** Earth sciences.
- **Github Copilot**: Earth sciences.
- **cohere:** Geology.

##### Prompt 7

```
prompt += f"""Topic entities: Keynes, Marx, Friedman, Smith, Hayek, Schumpeter, Malthus, Ricardo, Hegel, Adam Smith.
Topic label:"""

```

Outcomes:

- **chatGPT:** Economists and economic theories.
- **Github Copilot**: 19th and 20th century economists.
- **cohere:** Classical economists.

##### Prompt 8

```
prompt += f"""Topic entities: Relativity, Quantum mechanics, Electromagnetism, Thermodynamics, Astrophysics, Cosmology, Particle physics, String theory.

Topic label:"""

```

Outcomes:

- **chatGPT:** Physics theories.
- **Github Copilot**: Physics.
- **cohere:** 20th century physicists.

##### Prompt 9

```
prompt += f"""Topic entities: Shakespeare, Tolstoy, Dante, Chaucer, Austen, Hemingway, Whitman, Faulkner, Orwell, Camus.

Topic label:"""

```

Outcomes:

- **chatGPT:** Classic authors.
- **Github Copilot**: Authors.
- **cohere:** Writers and literary works.

##### Prompt 10

```
prompt += f"""Topic entities: Mona Lisa, Sistine Chapel, The Last Supper, The Starry Night, The Persistence of Memory, The Scream, The Kiss, The Dance, The Water Lilies.

Topic label:"""

```

Outcomes:

- **chatGPT:** Famous paintings and artists.
- **Github Copilot**: Paintings.
- **cohere:** Famous paintings.
