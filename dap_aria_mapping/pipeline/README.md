# Topic labelling

Three sets of labellings are possible:

- **entity labelling:** uses absolute counts of DBpedia tags across OpenAlex articles to label topics with the most frequent entities.
- **journal labelling**: labels topics of the hierarchical taxonomy by using a topic's most prominent associated academic journals, measured as how frequent entities in a topic are also entities in a journal. The ranking of relevant journals should account for the relevance of an entity within the topic, the relevance of an entity within the journal, as well as the relevance of a journal vis-à-vis other journals. To that end, we estimate the marginal contribution of journal $j$ to topic $c$ via the entity $i$ as:

  $$
  contrib_{(c,i,j)} = \log\left(2 + \text{freq}_{(c,i)}\right)^2\cdot\log\left(1 + \text{freq}_{(c,j)}\right)^2 \cdot \text{freq}_{(j,i)}
  $$

  where$\;\text{freq}_{(c,i)}$ is the frequency of entity $i$ relative to other entities in sub-topic $c$, $\;\text{freq}_{(c,j)}$ is the frequency of journal $j$ in sub-topic $c$ over other journals also present in $c$, and $\text{freq}_{(j,i)}$ is the frequency of entity $i$ in journal $j$ relative to other entities in $j$. For each subtopic $c$, scores for all journal contributions are then estimated, ie.

$$
scores_{(c,j)} = \sum_{i=1}^{I}contrib_{(c, i, j)} \quad \forall \;j \in \{1, 2, ..., J\}
$$

    and the top$\;n$ scores are used to build the labelling.

- **chatGPT** **labelling**: We leverage a reverse chatGPT API [wrapper](https://github.com/acheong08/ChatGPT#v1-standard-chatgpt), which uses session cookies from chat.openai.com to hit the API's endpoints directly. Through these, we can label the sets of entities as well as request confidence scores and a list of entities that GPT 3.5 considers "noisy" given the set of entities. These bots can be run in parallel and require instantiating a Chatbot class with a single argument (the session cookie). It also proxy tunnels connections to bypass firewalls and IP bans (such as OpenAI's for any AWS-related IPs).

  - This makes it much better at running in bulk in EC2 instances than the previous [wrapper](https://github.com/mmabrouk/chatgpt-wrapper), which uses real browsers and is thus limited in its ability to parallelize and dodge IP bans.
  - The outputs from the labelling are lists of four-item tuples, corresponding to topic ID, topic label, confidence score, and discarded entities, ie.

    `[('List 1', 'Machine learning', 100, ['Russian Spy', 'Collagen']), ('List 2', 'Cosmology', 90, ['Matrioshka', 'Madrid'])]`

To produce these labels as dictionaries, execute the following command.

`python dap_aria_mapping/pipeline/make_topic_name_assignments.py --taxonomy cooccur centroids imbalanced --name_type entity --levels 1 2 3 4 5 --n_top 5 --n_articles 50000`

Additional arguments include `show_count` (to include counts of entities in database), and `save` (to export jsons to S3).
