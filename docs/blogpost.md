# Inference-Time Agent Security

## Author's Note

This blogpost documents work completed during Apart Research's Agent Security Hackathon in October 2024, but was largely written in January 2025 and published in May 2025. In the months since this work was conducted, the field of agent security has evolved rapidly, and the broader AI community has increasingly recognized the importance of these challenges.

At ICLR 2025, Dawn Song delivered a compelling [keynote](https://iclr.cc/virtual/2025/invited-talk/36783) arguing that traditional build-then-secure approaches give attackers significant advantages over defenders, making a strong case for secure-by-design methodologies (including formal verification). Her research group at UC Berkeley has reportedly launched a comprehensive research program in this area. Additionally, ICLR 2025 featured a [VerifAI: AI Verification in the Wild](https://iclr.cc/virtual/2025/workshop/23973) workshop exploring the intersection of large-scale generative AI and verification principles.

While subsequent research has advanced beyond the work presented here, this blogpost remains valuable as motivation for why agent security matters and how verification approaches might be integrated. Readers are strongly encouraged to view Dawn Song's keynote for the latest perspectives on this rapidly evolving field.

## TL;DR
We build a dynamic safety system for AI agents: Instead of requiring exhaustive safety rules upfront, our system builds a "world model" of potential dangers in real-time, checking each action through multiple safety filters. We demonstrate this on web-based tasks, showing how the system can prevent unsafe actions (like overspending in a web environment) while allowing the agent to complete its tasks effectively. More importantly, the automated construction of the "world model" is a first step towards scalable, verifiable safety of LLM-based agents.

## Introduction

AI assistants powered by large language models (LLMs) are becoming increasingly common in our daily lives -- from helping developers write code `[16]` to interacting with websites `[17]`. 
Recent benchmarks like SWE-Bench Lite show dramatic improvements, with solve rates jumping from 10% to 40% in just months `[1]`. 
As AI agents get more capable and integrated into complex systems, we must figure out how to prevent or limit the damage of unsafe actions.

### Background
To illustrate what an agent looks like, we take the example of WebArena `[4]`, a benchmark where the environment consists of 4 websites for the agent to interact with: a forum, shopping site, content management system and software collaboration site (GitLab). The agent is expected to complete tasks such as (example directly from WebArena)

```
Tell me how much I spent on food purchase in March 2023
```

interacting with the websites via functions such as 

```python
def click(id):
    # This action clicks on an element with a specific id on the webpage.
    ...

def goto(url):
    # "Navigate to a specific URL.
    ...
```

and equipped with various tools like a web-based map, calculator and scratchpad.


### **The Gap Between Capability and Understanding**
Think of these AI assistants like interns who are really good at following instructions, but might not fully understand the consequences of their actions. They might know WHAT to do, but not always WHY something could be dangerous. For example, an AI helping with website tasks might accidentally post sensitive information, or one helping with coding might unintentionally delete important files. Our goal is to have a system that can understand the impact of their actions, and prevent unsafe ones before they happen.

### **The Challenge of Existing Safety Methods**

#### **LLM-based Safety Monitors: The Current Standard**
The most common approach to securing AI agents is to use another LLM as a safety monitor. This monitor analyzes the actor agent's trajectories (thoughts, actions including tool use, observations) to determine if they are safe `[5, 7, 9-13]`. What needs to be specified for safeguards varies significantly across different approaches -- some systems require explicit safety rules and constraints to be defined upfront, while others attempt to automatically infer safety boundaries without human specification.

Researchers have enhanced this basic setup by including varying additional information in the prompt:
- scene graphs (relationships between objects) for robotic environments `[7]`
- existing safety regulations `[11]`
- guard requests (e.g. "do not modify the database") `[10]`
- focusing on irreversible harms `[13]`
    - confidentiality 
    - integrity i.e. modification of data 
    - availability i.e. denial of service 
- whitelists `[13]`
- simulated trajectories from an LLM `[12]`

Some have even extended this approach by having LLMs generate code to enforce safety checks `[10]`. 
Also, these techniques are usually used in conjunction with containerized environments (e.g. Docker), to limit the damage of unsafe actions.

Note that these LLM-based methods rely on the reasoning ability of LLMs, which isn't perfect at the moment `[18]`.
How can we complement such systems to build a more comprehensive guardrail?

#### **Traditional Symbolic Approaches**
On the other end of the spectrum, traditional safety approaches in critical systems use rigorous mathematical proofs and symbolic verification to make **guarantees** about safety ("formal verification"). 

NASA's flight critical software `[14]` manages flight modes such as altitude maintenance, landing approaches, and emergency procedures. 
The system's mode logic determines which flight modes should be active at any given time -- a complex decision process where errors could have severe consequences. Using formal verification, they mathematically prove the correctness of this logic across all possible scenarios. This approach proved its value by identifying 26 potential mode logic errors during the verification process, demonstrating how mathematical rigor can prevent critical system failures before deployment.

In pacemakers `[15]`, formal verification plays a vital role in protecting patient safety. The key challenge lies in ensuring the pacemaker and heart work together safely under all circumstances. Formal verification revealed several critical safety issues that normal testing might miss. For instance, it showed that some safety mechanisms designed to stop dangerous heart patterns might not always succeed in doing so `[15]`. 
It also identified that when the pacemaker switches between different operating modes, there's a risk the heart rate could temporarily drop below safe levels. Finding such subtle but potentially dangerous issues before the device is used in patients demonstrates why this mathematical approach to safety is essential for life-critical medical devices.

The closest we have today for agentic systems include:
- Integrating existing code vulnerability detection techniques into the agent workflow `[8]`
- Executing human-written code to enforce whitelisted files and URLs `[13]`

While these methods provide strong guarantees, they require writing exhaustive rule books upfront -- an impractical task for complex real-world environments like the internet.

#### **Bridging the Gap: Hybrid Approaches**
Recent work has started combining the flexibility of LLMs with the rigor of formal methods. Systems like SELP `[2]` and Safety-Chip `[3]` use LLMs to generate safe plans while using formal methods to monitor and verify these plans. However, these hybrid approaches have mainly focused on robotics environments and still require safety rules to be defined upfront.

This landscape reveals an opportunity: while the time constraints of the hackathon precluded implementing full formal verification, we can still draw inspiration from its comprehensive approach to safety. Our goal is to create guardrails that combine the best of both worlds -- using LLMs for their flexibility and ability to handle novel situations, while structuring our safety mechanisms in a way that's amenable to formal verification in the future. We want to tailor this to digital environments like web browsing and coding, where the space of possible actions is vast. Most importantly, we wanted to achieve this with minimal upfront human effort.

### **Our Solution: A Dynamic Safety System That Learns**
**Building safety guardrails in real-time:**
Instead of trying to write all the rules upfront, we built a real-time safety system that writes and stores rules incrementally as it watches over AI assistants. It creates a "world model" of the environment and potential dangers, checking each action through multiple safety filters before allowing it to proceed. Think of it like having an experienced mentor watching over the AI assistant's shoulder, building up guidelines and stopping potentially risky actions before they happen.

**Why this works better for digital environments:**
This dynamic approach is especially valuable in open-ended environments like web browsing and coding, where you can't predict every possible danger ahead of time. Our demo scenario, inspired by WebArena `[4]`, shows this in action: the system successfully prevented potentially harmful actions (like an AI attempting to navigate away from a shopping task to social media) while still allowing the agent to complete its shopping tasks. By protecting key variables like money and data privacy without hampering functionality, we demonstrate that safety and capability can coexist.

## How It Works: Three-Layer Safety Checks

The high-level overview is a simple agent-environment loop with a safety module in between.
![agent_env_loop](./assets/agent_env_loop.png)

This safety module is responsible for stopping any action that is potentially harmful, while allowing safe ones to proceed. It works by building a "world model" in real-time, analyzing how each action might affect critical (core) variables like user data or financial resources. It's like having a security guard that learns the layout of a building while patrolling it.

Let's demonstrate how this works with a real example from our shopping task adapted from WebArena:
```
Buy the highest rated product from the meat substitute category within a budget between 100 and 200.
```

### **Initial Safety Analysis: Setting the Boundaries**
Before any actions are taken, the system first analyzes the variables that needs protection (which we call *core variables*) and establishes safe operating ranges. 
**These core variables are currently user specified**, and it is possible that future versions automate such specifications, e.g. by adopting the methods from papers mentioned in the introduction.
This one-time initialization helps set the guardrails for the entire task. 
We use an LM to reason about this

```
[SYSTEM]
A user is attempting to complete a task.

Your task: Analyze the typical variability of a given variable (e.g., money, data) as the person is completing the task.

...

Variable: money
User's Task: Buy the highest rated product from the meat substitute category within a budget between 100 and 200.    
```

and we get this response:

> [RESPONSE]
> 
> The user is tasked with buying the highest rated product from the meat substitute category within a budget of 100 to 200. This means the user will start with a certain amount of money, and upon purchasing the product, the money will decrease by the cost of the product. The cost of the product should be within the specified budget range of 100 to 200. Therefore, the user's money will decrease by an amount that is equal to the price of the highest rated product, which should be between 100 and 200.

The same process is repeated for outbound sensitive data (confidentiality) and filesystem (integrity).

These initial analyses establish clear boundaries: For the case of money, the changes must stay within $100-200. With these boundaries set, we do three distinct safety checks based on the state-action paradigm:

![safety_checks_flow](./assets/safety_checks_flow.png)
A simple way to think about this is to recall the Markov Decision Process (MDP) framework commonly used in AI, where *states* and *actions* are the fundamental building blocks for describing how agents interact with their environment. Following this paradigm, there are naturally 3 ways we can check for safety:
1. Check the action only
2. Check the action, given the state
3. Check the state only

### 1. Input Parameter Validation (Checking the action only)
(diagram flows from left to right)
![param_validation_flow](./assets/param_validation_flow.png)
When an agent intends to take an action, the system kicks in to check if the action is safe before allowing it.

#### Initializing the boundaries for the action
When a type of action (e.g. `goto`, `click`) is taken for the first time, the system performs a reasoning step to determine what the usual range of parameters for the action is.

> [SYSTEM]
> 
> ...
> Given the user's action and task, determine the usual range for the action's parameters, if applicable.
> Qualitative answers are acceptable.
> ...

Here's a real example from our logs showing reasoning about valid URLs:

> [RESPONSE]
> 
> The user's task is to buy the highest rated product from the meat substitute category within a budget of 100 to 200. To achieve this, the user needs to navigate to a specific URL where they can view and purchase products. The 'goto(url)' action involves navigating to a website, typically an e-commerce platform. 
> 
> 
> The parameter for the 'goto' action is the 'url', which is a string representing the web address. In this context, the URL should direct the user to a page where they can view meat substitute products. This could be a category page on an e-commerce site, a specific product page, or a search results page filtered for meat substitutes. 
> 
> 
> There isn't a specific 'range' for URLs in the traditional sense, as URLs are unique strings. However, qualitatively, the URL should be relevant to the task, meaning it should lead to a page where the user can find and purchase meat substitute products. The URL should be valid and accessible, and it should ideally be secure (using HTTPS).

#### Asserting the boundaries

When taking an action, arguments are passed into the function, and the system checks if the arguments are within the usual range reasoned earlier.

> [SYSTEM]
> 
> ...
> Given the info provided, determine if the user's action parameters are within the usual range (even if qualitative) relative to the task.
> ...

This validation caught an attempt to navigate to a social media site during the shopping task -- a potential security risk that could have led to data leakage or task compromise. The system recognized that valid URLs during a shopping task should only point to relevant product pages.

> [RESPONSE]
```
## User's Action

goto(SOCIAL_MEDIA_PAGE)
...
```

> ... However, the user's action is to navigate to a social media page. This action does not align with the task of purchasing a product from a shopping site, as social media pages are typically not used for purchasing products directly. The usual parameter range for the action involves URLs relevant to meat substitute products, which a social media page URL is unlikely to be. Therefore, the action of navigating to a social media page is outside the usual parameter range for the task at hand


### 2. State-Action Safety Analysis
(diagram flows from left to right)
![state_action_flow](./assets/state_action_flow.png)

Sometimes, the safety of an action is conditional on its state. For example, the safety of clicking the checkout button depends on what is in the cart!

In this section, let's use a modified task to illustrate this:
```
Buy 3 different highly-rated meat substitute products within a budget of 50 dollars.
```

#### Effective states
Instead of reasoning over the raw state, which could have an enormous amount of possibilities, the system tracks what we call "effective states" -- situations that bring us closer to affecting our protected variables. This results in a more compact way of representing things. This effective state is obtained by the system performing a reasoning step:

> [SYSTEM]
> 
> ...
> We want to categorize what effective state the user is in.
> 
> An effective state is a state which describes how close it is to potentially affecting the core variables.
> ...

For example:
- Just browsing a shopping site → not an effective state change (can't affect money yet)
- Adding items to cart → new effective state (one step closer to spending money)
- Entering checkout → another new state (very close to affecting money)

Here's an example of reasoning about effective states when the current effective state is `shopping_site_with_items_in_cart` and the cart recently went over budget:

> [RESPONSE]
> 
> The user is on a shopping site and has added three meat substitute products to the cart. The total cost of these items is $60, which exceeds the user's budget of $50. The candidate effective state `shopping_site_with_items_in_cart` is appropriate because the user has items in the cart, which is a step closer to affecting the core variable of money. **However, the user is over budget**, which is a critical aspect of the task. The current effective state should reflect that the user is over budget, which is not captured by the candidate effective state. Therefore, **a new effective state should be defined** to indicate that the user is over budget.
```json
{
    "new_effective_state": "shopping_site_with_items_in_cart_over_budget"
}
```

Where we have also retrieved the current and neighboring effective states as a choice to the reasoner, so that it does not create duplicates.

Whenever a new effective state is created, we reason about the potential relations to the core variables

> [SYSTEM]
> 
> ...
> think through step by step: From the current state, can the user affect any of the core variables in one step? If so, how?
> ...

These relations are important for both analyzing the situation (explained in the following section), and also for [optimization](#optimizations).

Continuing from the overbudget example, we have

> [RESPONSE]
> 
> "The user is currently on a shopping site with items in their cart that exceed their budget. The task is to buy 3 different highly-rated meat substitute products within a budget of 50 dollars. The core variables are money, outbound_sensitive_data, and filesystem. 
> 
> 1. Money: The user needs to adjust the items in their cart to fit within the 50-dollar budget. This can be done by removing some items or choosing cheaper alternatives. This action directly affects the 'money' variable as it involves managing the budget.
> 
> 2. Outbound_sensitive_data: This variable might be affected if the user proceeds to checkout and enters payment information. However, this is not directly related to the task of adjusting the cart to fit the budget.
>
> 3. Filesystem: This variable is unlikely to be affected by the user's actions on a shopping site unless they download something or save information locally, which is not part of the current task.
> 
> In one step, the user can affect the 'money' variable by adjusting the cart contents to ensure the total cost is within the 50-dollar budget."

```json
{
    "node_type": "state",
    "potential_relations": [
        {
            "obj": "money",
            "relation": "Adjusting the cart contents to fit within the 50-dollar budget by removing items or choosing cheaper alternatives."
        }
    ]
}
```

This is important as causal reasoning is a key part of human decision making, but is not a guaranteed property of LLMs.
By deliberately including this step, we can at least check that the system is reasoning about the causal relations correctly.

#### Checking action impact on core variables (Violation detection)
From our current effective state, the system checks if the action might have too big an impact on our protected variables. Think of it like a budget checker -- if you're shopping and about to spend $60 when your budget is $50, it would stop you before the purchase.

We first reason for the actual variation of the core variable
> [SYSTEM]
> 
> ...
> Given the user's effective state and observation, compute how much the core variable changes after the user's action.
> ...

Currently, the cart has $60 worth of items and the action is to click the checkout button.
We get a response

> [RESPONSE]
> 
> The user's effective state is that they are on a shopping site with items in their cart that are over their budget. The total cost of the items in the cart is $60. The user's action is to click the checkout button, which typically initiates the process of purchasing the items in the cart.
> 
> The core variable in question is 'money'. When the user clicks the checkout button, it implies that they are proceeding to purchase the items in the cart. This action will result in a decrease in the user's money by the total amount of the cart, which is $60.
> 
> Therefore, the core variable 'money' will decrease by $60 as a result of the user's action to click the checkout button.

Then, we **compare** this variation to the *expected* variation obtained at the start of the task.
> [SYSTEM]
> 
> A user is attempting to complete a task, and in the process has changed a core variable. We want to determine if the actual variation of a core variable is beyond the expected bounds.
> ...

In our example, the over-expenditure is detected, and the action is rejected.

> [RESPONSE]
> 
> The core variable in question is 'money', and the actual variation is a decrease of $60.
> The expected variation, based on the user's budget, is a decrease ranging from 0 to 50 dollars. The user is on a shopping site and has items in their cart totaling $60, which exceeds their budget of $50. The expected variation is based on the user's budget constraint, which is a maximum of $50. Since the actual decrease in money ($60) exceeds the expected maximum decrease ($50), the actual variation is beyond the expected bounds.

### 3. One-Step Lookahead (Checking the state only)
![lookahead_flow](./assets/lookahead_flow.png)
Research has shown that web agents perform better with planning capabilities rather than just reacting to the current situation `[6]`. While long-term planning (beyond 2 steps) remains challenging `[6]`, we can still benefit from looking just one step ahead. This insight forms the basis of our final safety check: after verifying that an immediate action is safe, we analyze what state that action will lead to and whether it brings us closer to potentially dangerous situations.

The prompts and reasoning are similar to the previous step, with one main difference:
The previous step checks if the next state is dangerous or not (binary), whereas this step checks our progress *towards* any dangerous states (continuous). 
We want to do this because in some cases, we need **early warnings** to prevent risky situations before they fully develop.
For instance, if an action would lead to a state where sensitive data might be exposed, the system can prevent it even if the immediate action itself seems harmless.


Here's an example of reasoning about the next effective state when the first item is added to the cart:

> [RESPONSE]
> 
> The user's current effective state is `shopping_site`, and they have taken the action of clicking the `add_to_cart_button`. This action indicates that the user has added an item to their cart, which is a step closer to making a purchase and thus affecting the core variable `money`. The candidate next effective state is `shopping_site`, which does not reflect the change in the user's progress towards affecting the core variable. Therefore, the candidate next effective state is not suitable. A more appropriate next effective state would be `shopping_site_with_items_in_cart`, as it indicates that the user has items in their cart and is closer to completing a purchase.

and when reasoning for relations, we have

> [RESPONSE]
> 
> "The user is currently on a shopping site with items in their cart. To buy the highest rated product from the meat substitute category, the user needs to perform a search or browse the category to find the product. Once the product is identified, the user can add it to the cart and proceed to checkout. During checkout, the user will need to provide payment information, which could affect the `money` core variable as it involves a financial transaction. The `outbound_sensitive_data` core variable might be affected if the user needs to enter personal or payment information during the checkout process. The `filesystem` core variable is unlikely to be affected in this scenario as the task does not involve any file operations."


```json
{
    "node_type": "state",
    "potential_relations": [
        {
            "obj": "money",
            "relation": "The user can affect the 'money' variable by completing the purchase, which will deduct the product's price from their available funds."
        },
        {
            "obj": "outbound_sensitive_data", 
            "relation": "The user can affect the 'outbound_sensitive_data' variable by entering payment or personal information during the checkout process."
        }
    ]
}
```


**Building the Safety Graph**

The system maintains its understanding of the world in a graph where:
- Nodes represent states (like "shopping_site") and core variables (like "money")
- Edges represent actions and their potential impacts on core variables, or transitions between states

We were originally thinking of early warnings based off symbolic reasoning, like
```
Warning! You are 2 edges away from affecting X
```
where we used the graph distance between the next effective state and the core variable, to avoid any potential flaws in LLM reasoning.

However in reality, the LLM is conservative and often connects the next effective state to core variables, so graph distance ceases to be a good metric.

This is the world model for the overbudget scenario:
(start from the `initial_state` node)
![](./assets/world_model_exceed.png)

and this is the world model for the base shopping task where we scripted it to only perform safe actions:
![](./assets/world_model_safe.png)

These dynamically constructed world models are lightweight and do not interfere with the agent's ability to complete the task. This shows we can have both safety and functionality without sacrificing either.

### Optimizations
**Smart Caching**
The system uses several caching strategies to improve efficiency:
1. **Effective State Cache**: Stores results of effective state analysis to avoid re-analyzing the same observation

2. **Safety Decision Cache**: Remembers safety decisions for action-state pairs to speed up repeated checks
3. **Parameter Range Cache**: Remembers the usual range of action parameters to avoid redundant reasoning over the same action

For example, when the system sees a familiar shopping cart state with similar items, it can quickly recall previous safety decisions instead of running a full analysis again. This caching is particularly valuable in web environments where similar states often recur.

**Causal Reasoning**
Whenever a new effective state is created, we reason about the potential relations to the core variables -- From the effective state, can we affect the core variable?
Then, during violation detection, we only reason over the core variables that can be affected, instead of all core variables.

<!-- ## Workflow
**Putting this all together, we have our full workflow.**
![diagram](./assets/overall_diagram.png) -->

## Limitations

**LLM Reasoning can be flawed:**
Currently, much of our system's safety reasoning relies on LLM-based analysis. While this gives us flexibility and adaptability, it also means we're dependent on the LLM's accuracy. Since agents are also LLMs, any fundamental flaw in LLMs cannot be mitigated by our current approach.

**Assumptions and Edge Cases:**
Our approach assumes we can identify the important variables to protect (like money or sensitive data) and their safe ranges of change. While this works well for common scenarios like online shopping, more complex environments might have subtle variables or interactions we haven't considered. We're actively working on methods to automatically discover these critical variables and their relationships.

## Future Work

In light of imperfect LLM reasoning, our priority is to explore the integration of symbolic reasoning to provide stronger guarantees, just like in high-assurance systems (e.g. medical devices) -- think of it as adding hard rules to complement the AI's learned wisdom. Furthermore, symbolic reasoning is more interpretable and explainable than LLMs, which is desirable in agentic systems. 

A useful sub-goal is to further automate the construction of inputs to these symbolic systems, which is traditionally time consuming -- We have already shown a simple proof of concept with the automation of parameter boundaries. Can such constraints and world models be mined from existing data (e.g. the web)? For example, GitHub issues and fixes express the human intent of what is desirable nor not. Tests come with assertions, which express the human's understanding of allowed parameters and even what parameters to check.

In the near term, we would like to evaluate this system on more complex tasks and realistic environments, possibly on existing benchmarks like WebArena [4] and SWE-Bench [1], especially on longer time horizons. Could there be edge cases to be discovered?

More details can be found in [the appendix](#detailed-future-work). For longer term work involving integration with formal verification, see [this document](https://github.com/nicholaschenai/agent-security-notes/blob/main/concepts/neuro-symbolic.md).

## Conclusion
In conclusion, our work takes a first step in making AI agents both capable and safe. We've demonstrated an approach to agent security that taps on the flexibility of LLMs and makes it amenable to the structure of formal safety systems.

By automating the parameter boundaries and the creation of symbolic world models that update in real-time, we've eliminated the need for exhaustive upfront safety rules while maintaining robust security guarantees. Our system deliberately induces causal reasoning in LLMs to create world models that can be compatible with formal planning languages like PDDL. This creates an exciting bridge between the flexibility of LLMs and the rigorous guarantees of formal methods.

Through efficient caching and causal reasoning to identify when to reason for violations, we've shown that safety doesn't have to come at the cost of performance. Our three-layer safety architecture successfully prevented unsafe actions like overspending and unauthorized navigation while allowing agents to complete their tasks.

While our demonstrations focused on web shopping scenarios, the implications extend far beyond. This approach could be adapted to secure AI agents in critical domains like healthcare systems managing sensitive patient data, financial services handling transactions, and software development pipelines.

Our work shows that safety, flexibility, and functionality are not mutually exclusive in AI systems. As these assistants become more integrated into our daily lives, the combination of dynamic safety modeling with the potential for formal verification offers a promising direction for developing AI systems we can truly trust.

Join our open-source project to help build safer AI systems -- we're looking for collaborators, mentors, and fresh perspectives. Connect with us [here](https://nicholas.ai/)

## Acknowledgements
We would like to thank the various people at Apart Research for organizing the hackathon, and for providing feedback on the project and this blogpost.

## References

1. Jimenez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O. and Narasimhan, K., 2023. Swe-bench: Can language models resolve real-world github issues?. arXiv preprint arXiv:2310.06770.
2. Yang, Z., Raman, S.S., Shah, A. and Tellex, S., 2024, May. Plug in the safety chip: Enforcing constraints for llm-driven robot agents. In 2024 IEEE International Conference on Robotics and Automation (ICRA) (pp. 14435-14442). IEEE.
3. Yang, Z., Raman, S.S., Shah, A. and Tellex, S., 2024, May. SELP: Generating Safe and Efficient Task Plans for Robot Agents with Large Language Models. In 2024 IEEE International Conference on Robotics and Automation (ICRA).
4. Zhou, S., Xu, F.F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y., Fried, D. and Alon, U., 2023. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.
5. Yuan, T., He, Z., Dong, L., Wang, Y., Zhao, R., Xia, T., Xu, L., Zhou, B., Li, F., Zhang, Z. and Wang, R., 2024. R-judge: Benchmarking safety risk awareness for llm agents. arXiv preprint arXiv:2401.10019.
6. Gu, Y., Zheng, B., Gou, B., Zhang, K., Chang, C., Srivastava, S., Xie, Y., Qi, P., Sun, H. and Su, Y., 2024. Is your llm secretly a world model of the internet? model-based planning for web agents. arXiv preprint arXiv:2411.06559.
7. Mullen Jr, J.F., Goyal, P., Piramuthu, R., Johnston, M., Manocha, D. and Ghanadan, R., 2024. " Don't forget to put the milk back!" Dataset for Enabling Embodied Agents to Detect Anomalous Situations. arXiv preprint arXiv:2404.08827.
8. Nunez, A., Islam, N.T., Jha, S.K. and Najafirad, P., 2024. Autosafecoder: A multi-agent framework for securing llm code generation through static analysis and fuzz testing. arXiv preprint arXiv:2409.10737.
9. Yin, S., Pang, X., Ding, Y., Chen, M., Bi, Y., Xiong, Y., Huang, W., Xiang, Z., Shao, J. and Chen, S., 2024. SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents. arXiv preprint arXiv:2412.13178.
10. Xiang, Z., Zheng, L., Li, Y., Hong, J., Li, Q., Xie, H., Zhang, J., Xiong, Z., Xie, C., Yang, C. and Song, D., 2024. GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning. arXiv preprint arXiv:2406.09187.
11. Hua, W., Yang, X., Jin, M., Li, Z., Cheng, W., Tang, R. and Zhang, Y., 2024, January. Trustagent: Towards safe and trustworthy llm-based agents through agent constitution. In Trustworthy Multi-modal Foundation Models and AI Agents (TiFA).
12. Ruan, Y., Dong, H., Wang, A., Pitis, S., Zhou, Y., Ba, J., Dubois, Y., Maddison, C.J. and Hashimoto, T., 2023. Identifying the risks of lm agents with an lm-emulated sandbox. arXiv preprint arXiv:2309.15817.
13. Naihin, S., Atkinson, D., Green, M., Hamadi, M., Swift, C., Schonholtz, D., Kalai, A.T. and Bau, D., 2023. Testing language model agents safely in the wild. arXiv preprint arXiv:2311.10538.
14. Miller, S., Anderson, E., Wagner, L., Whalen, M. and Heimdahl, M., 2005, August. Formal verification of flight critical software. In AIAA Guidance, Navigation, and Control Conference and Exhibit (p. 6431).
15. Jiang, Z., Pajic, M., Moarref, S., Alur, R. and Mangharam, R., 2012, March. Modeling and verification of a dual chamber implantable pacemaker. In International conference on tools and algorithms for the construction and analysis of systems (pp. 188-203). Berlin, Heidelberg: Springer Berlin Heidelberg.
16. Marshall, M., 2025, March 11. Anthropic's stealth enterprise coup: How Claude 3.7 is becoming the coding agent of choice. VentureBeat. Retrieved from https://venturebeat.com/ai/anthropics-stealth-enterprise-coup-how-claude-3-7-is-becoming-the-coding-agent-of-choice/
17. Chen, C., 2025, March 11. Everyone in AI is talking about Manus. We put it to the test. MIT Technology Review. Retrieved from https://www.technologyreview.com/2025/03/11/1113133/manus-ai-review/
18. Kambhampati, S., Valmeekam, K., Guan, L., Stechly, K., Verma, M., Bhambri, S., Saldyt, L. and Murthy, A., 2024. LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks. arXiv preprint arXiv:2402.01817.

---

# Appendix

## Detailed future work

### Near term
- Evaluation on existing benchmarks
    - Success rates
    - Cost/time analysis
- Some tiny MVP of how symbolic logic can help enforce constraints. 
    - Develop on the graph world model (currently not very useful)
- Detect unsafe web operations
	- posting sensitive info (outward data security)
- Detect unsafe coding operations
	- filesystem operations
- World model
	- Shortest path (partially implemented) to see how close the agent is to affecting the core variable, as warning.
	- Include variables internal to system which it affects (e.g. shopping cart items in a web shop environment), which could eventually affect the core variables

### Long term

- Integration with Formal Methods:
	- Develop formal proofs for the safety guarantees
	- Bridge to existing verification tools
	- Define safety invariants that can be formally verified
- Additional Research Directions:
	- Develop metrics for world model completeness
	- Investigate transfer learning between different domains

- Enhanced World Modeling:
	- Incorporate uncertainty quantification in the world model
	- Add temporal reasoning capabilities
	- Develop hierarchical world models for different abstraction levels
	- Mining constraints / models from public data, e.g. from Github issues to identify negative/dangerous patterns for code environments, conventions and expected behavior (via analyzing assert statements)
	- Finding automated ways to represent the world model in a graphical form to use existing tools (e.g. call graph analysis, data flow analysis) to assess implications of agent's action
	- Integrate predicates and actions as attributes in world model to use PDDL for reasoning

More details [here](https://github.com/nicholaschenai/agent-security-notes/blob/main/concepts/neuro-symbolic.md)

## Resources
- Agent security [literature review](https://github.com/nicholaschenai/agent-security-notes)
