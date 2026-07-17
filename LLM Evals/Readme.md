# LLM Evaluation
![](https://cdn.sanity.io/images/k7elabj6/production/6fe91ce5a30a3a26401151ae1ec13e014e816f92-1536x1024.png )
LLM Evals are systematic, repeatable tests used to judge an LLM or LLM-powered system against a clear criteria. LLM Evals is not just a metric. It is basically the entire testing setup.  
**Two Types of LLM Evaluations:**
* Model Evals: "These evaluate the model itself... the main idea is to test and evaluate the capabilities of a model."
* Application Evals: "These are the ones used to evaluate LLM-based applications where the LLM is just one component."
While Model Evals (benchmarks) are important for decision-making, the primary focus for engineers is Application Evals because "your job as an AI engineer is to build LLM-based applications, so to evaluate [them] is also your job.

# End-to-end workflow for evaluating LLM-based applications

* Define Task and Target: Clearly establish what the system must achieve (e.g., accurate classification).
* Define Success Criteria and Metrics : Select measurable indicators, such as Accuracy (the percentage of emails correctly routed).
* Build a Golden Dataset : Curate a labeled dataset of 50–500 real-world examples (human-verified) to serve as your testing benchmark.
* Choose an Evaluation Method : Determine how tests will run. Methods can be Automated (code-based scripts), Human-led (manual review), or LLM-based (using another model to grade outputs).
* Run the Model: Pass your golden dataset through the system.
* Evaluate and Analyze Results: Compare the outputs against expected labels to calculate performance and pinpoint where the model fails.
* Improve the System: Optimize the system prompt or upgrade the underlying LLM based on error patterns.
* Iterative Loop: Re-run the evaluation cycle to confirm that changes have led to measurable improvements.
* Deployment and Production Monitoring: Deploy the model while maintaining a feedback loop. When the system fails in production, capture those specific cases and add them to your Golden Dataset to prevent future regressions.

# Why our AI Application Needs Multiple Eval Pipelines?
Our individual components of a Retrieval-Augmented Generation(like Retriever and model) might be working fine but as a whole AI Application can still fail. For eg. we had chose k=5 in retriever, but 4 irrelevant chunks can corrupt the query to our model and model might get confuse.  
So we should have a combined evaluator(to test our application combined along with individual evals at each step)
Along with accuracy, we also need to maintain low latency, optimal cost per query, and data privacy(We should have strict guardrails to avoid revealing other users' information or any organization-specific data that the requesting user is not authorized to access).
**The Three Pillars of Risk:** 
Evaluation pipelines should be structured around three risk categories:
* Application Quality: Ensuring the model provides correct, relevant, and complete answers.
* Safety: Guarding against toxicity, data leaks, and jailbreak attempts.
* Operations: Managing performance, including latency under load and cost per request.

## Programmatic/Deterministic Pipeline
### Recall@k For Retriever :
**Recall@K** Out of all the correct documents that exists, how many did the system retrieve in its top k results.(Number of relevant items in top k/Total number of relevant items)   
Create a Golden test dataset(by human or a powerful intelligent model - one time cost) , recall@k will be mean of all recalls from each query in this dataset.
### How can we improve?
* Fixing the embedding model.
* Fixing the input query with help of LLM.
* Increase K.
* Reranking(reranking the top k results according to query).

## Human Evaluation
* Red teaming: attacks LLM systems for testing jailbreaking, prompt injections, edge-case scenarios, and tries to make model produce harmful/wrong output.
* A/B Testing: real users as grader in the production(can be evaluated as thumbs-up, compare two responses)
* Direct grading/rating: Person reads outputs and scores them against a rubric(the Eval criteria on the basis of pipeline we had created)
* Gold answer/Dataset creator: Human creating the golden dataset.
* HITL(Human in the loop): Human reviews/rejects/approves/edits live outputs.

## Two types of LLM evaluation
* Reference Based: A known correct answer written down in advance for each test case, we grade by comparing output against that reference
* Referece free: We have no predefined correct answer, we judge the quality directly on its own terms, against a criteria/rubric but a rubric here is a scale/standard("what does a good answer looks like and not a per-term correct answer).

# Offline vs Online Evals
## Offline Evals
offline evals helps us do pre-release testing(gating), version comparison of two versions of agent/system prompt/models and regression testing(if we try to improve one thing in complex system, then other thing might get affected so try to have all types of cases in golden dataset).

## Online Evals
1. Unanticipated inputs(questions mixing, different languages, ambiguous questions, angry rants, prompt injection, edge-cases)
2. Emergent/systemic failures(problems that exist at scale, under load, over time(p99 latency), a subtle bias across a user group)
3. Data Drift(changes in documents, data distribution) so eval should also be updated accordingly.

### Online Evals pipeline:
Try to log things(maybe you can delete in a week), what you will log:
- Identification: Conversation Id, user Id, timestamp
- Input: raw input plus any preprocessing(normalized text/detected intent by model)
- Retrieved context
- Output: model name/version id, prompt version, tool calls, finish reason.
- Operational telemetry: derived cost, tokens, latency(separtely store retrieval and generation latency),
- Downstream user signals: user behavioural metrics(thumbs up/down, check if user is getting frustrated, user wanted an escalation, repeated query). Langsmith is a tool where you can store this.

Logging should be non-blocking(should not increase latency), durable+queryable, late-signal attachment(signals lands at different times), PII handling(scrub/tokenize Personally Identifiable Information like emails, phone numbers,, payment details, apply retention limits and access control). 
Langsmith is a complete platform to manage and track all our experiementations of chatbots/agents.

# LLM Benchmarking
To understand Benchmarks, first we need to have a clear understanding of LLM capabilities
LLM Capabilites:
1. Knowledge and Reasoning: Factual recall on subjects(**MMLU(Massive Multitask Language Understanding)** benchmark evaluates LLMs on 57 subjects.
2. Coding and software engineering: Highest revenue from LLM is coming from this field only(Claude has only 20Mn monthly active users with revenue of $40Bn while ChatGPT has 1 Biilion Weekly active users with revenue $25Bn)
  - Function-level code generation
  - Real world bug fixing in existing codebases
  - Multi-file, long horizon tasks
  - CLI level installation, configuration
  - API and function calling
3. Mathematics (Testing LLMs on Grade school to research level mathematical reasoning)
4. Long context (Single fact retrieval from long contexts, aggregation over long content, code repository understanding)
5. Vision and Multimodal capabilities
6. Agentic and tool use
7. Safety and Alignment(to jailbreaks, prompt injection, cybersec capabilities)
8. Instruction following
