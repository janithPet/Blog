---
layout: post
title:  "Literature Review"
date:   2016-10-30 01:59:21 +0100
categories: jekyll update
---

Literature Review: A Taxonomy of Reinforcement Learning
=======================================================

Machine learning can be thought of as subset, of artificial intelligence
that makes use of statistical methods to allow machines to show features
of learning, such as the inference of functional relationships between
data [@ml-girolami], and improvement with experience [@ml-mitchell].
There is often considered to be three main categories of machine
learning: supervised learning, unsupervised learning and reinforcement
learning [@2005-lawrence]. In supervised learning, structure is often
given to data before the learning exercise, for example in the form of
an input and an output; in imitation learning for example, the agent is
provided with examples of strategies and policies that have already been
deemed to be good by its ’teacher’ [@2013-kober]. In unsupervised
learning, there is no structure given to the data, and the learning
exercise attempts to infer or impose structure through its algorithms.
Reinforcement learning can be thought to sit in the middle, where
unstructured data is given structure through an iterative interaction
with an ’environment’ such that a certain goal is reached through the
maximisation of a reward [@2005-lawrence; @rl-sutton; @2010-deisenroth].

Due to its relevance for this project, I will be presenting the latest
research that has been done with regard to reinforcement learning in the
rest of this review. First, the main flow diagram of reinforcement
learning will the expressed, followed by a taxonomy of different methods
developed for completing each of the components of the flow diagram.
Finally, I will be presenting interesting research that has been carried
out with regards to the use of Gaussian Process Latent Variable Models
(GP-LVMs) in the context of reinforcement learning, and why it will form
the main focus of this project.\

What is Reinforcement Learning?
-------------------------------

Reinforcement learning (RL) consists of an *agent* interacting with its
*environment* by carrying out *actions*. The agent then receives signals
(which include the environment’s new *state* and *reward*) back from the
environment which it then uses to select the action it will make next,
such that the reward is maximised over time. This is illustrated in
Figure \[fig:rl-diagram\].

More formally, the agent receives a representation of the environment’s
states at time $t$, $s_t \in S^+$ where $S^+$ contains all the possible
states that the environment can take. Based on this current state, the
agent selects an action $a_t \in A(s_t)$, where $A(s_t)$ contains all
the possible actions that the agent can take while being in state $s_t$.
The mapping between the state $s_t$ and the action $a_t$ is called the
agent’s policy. Conventionally this is denoted by $\pi_t$. $\pi_t$ can
be deterministic, where $a_t = \pi_t(s_t)$, or probabilistic, the action
is selected from a probability distribution defined by $\pi(a_t|s_t)$.
Once $a_t$ is enacted, the environment will provide the agent with the
next state $s_{t+1}$, along with a reward $r_{t+1}$. The main goal of
RL, or the Rl problem then is to find the policy that will maximise the
long term sum of rewards $R_t$ (Equation \[eq:total\_r\])
[@rl-sutton; @2013-kober]. This is called the optimal policy $\pi^*$; RL
algorithms attempt to find $\pi^*$ using the reward signals by changing
its policy as it gains experience [@rl-sutton].
$$\label{eq:total_r} R_t = r_{t+1} + r_{t+2} + ... +r_T$$ In order to do
this, the agent must be able to evaluate how *good* it is to be in a
certain state with the current policy in terms of the expected long term
rewards. In other words, the *value* of being in state $s$ is the long
term reward that is expected by following the current policy after
starting from $s$ [@rl-sutton; @2016-kaiser]. This is given by Equation
\[eq:state\_value\]. This is called the *state-value function*.
$$\label{eq:state_value} V^\pi(s) = E(R_t|s_t = s) = E(\sum_{k=0}^{K}{\gamma^kr_{t+k+1}|s_t=s})$$
where $\gamma$ is the discount rate. It measures how much into the
future the agent looks at; if $\gamma = 0$, the agent is myopic, and if
$\gamma = 1$, the agent looks at all future rewards equally. Similarly,
the *action-value function* provides the expected long term reward for
being in state $s$ and following action $a$ given the current policy. It
is given by Equation \[eq:action\_value\].
$$\label{eq:action_value} Q^\pi(s,a) = E(R_t|s_t = s,a_t=a) = E(\sum_{k=0}^{K}{\gamma^kr_{t+k+1}|s_t=s, a_t=a})$$

The framework of RL provides great flexibility in terms of where it can
be applied to. The time step does not necessarily need to relate to
fixed intervals of time, but can often be individual steps in a decision
making process [@rl-sutton]. Furthermore it is often the case that no
prior task-specific knowledge is needed; through the process of trials,
the algorithm gains the experience it needs to have to find the best
path to achieve its goal [@2010-deisenroth], and it does this without
the presence of a teacher [@2016-kaiser]. It is also important to note
that RL is capable of being applied to problems to which classical
machine learning and other solutions to simpler problems have been
applied to. It is capable of solving problems that require both
interactive sequential prediction as well as the assimilation of complex
reward structures; most other methods are only able to reconcile a
spectrum of one [@2013-kober].

### Key assumptions in Reinforcement Learning

Before moving on, there are some key concepts that need to be addressed
in the context of RL.

-   The first of these is the agent-environment boundary. The boundary
    between the two does not necessarily need to be physical
    [@rl-sutton]. More often than not, the agent is simply an abstract
    construct that is the decision maker, and the physical systems that
    it interacts with then form the environment. For example, in the
    case of a robotic arm, the environment would consist of all the
    motors and the structural components of the arm, while the agent
    would simply be the ’brain’ that decides what torque to provide the
    motor with.

    A general rule that can be followed is that the agent-environment
    boundary is the limit of the agent’s absolute control [@rl-sutton];
    if the agent cannot arbitrarily change something, it is to be a part
    of the environment. It should be noted however that this does not
    form the limit of the agent’s knowledge. That is, the agent can,
    potentially know everything about its environment, and yet not be
    able to control everything [@rl-sutton].

-   <span>A common assumption in RL algorithms is that the states of
    system have the Markov Property. This property states that knowledge
    of the current state is sufficient to make decisions that also need
    to take into account all the historical states that were necessary
    to be at the current state. Mathematically speaking, a state is said
    to be Markov if and only if the one step prediction given by
    Equation \[eq:full\_state\] can be made by simply using Equation
    \[eq:markov\_state\].
    $$\label{eq:full_state} P(s_{t+1} = s^\prime, r_{t+1} = r^\prime | (s_t,a_t,r_t), (s_{t-1},a_{t-1},r_{t-1}), ... ,r_1),(s_0,a_0))$$
    $$\label{eq:markov_state} P(s_{t+1} = s^\prime, r_{t+1} = r^\prime | (s_t,a_t))$$
    This assumption is powerful as it allows for a RL algorithm to be
    made simpler, as it only needs to take into account the current
    state to predict the next state and reward when searching for the
    optimal policy. A reinforcement learning task that consists of
    Markov states is called a Markov Decision Process (MDP)
    [@rl-sutton].</span>

### The many faces of Reinforcement Learning

Due to the generality of RL and its applications, there are many ways by
which the different algorithms that have been developed in light of it
can be categorised. These categorisations are shown in Figure
\[fig:rl-classification\].

#### Finite Horizon vs Infinite Horizon problems

Simply put this distinguishes between the case where $T$ in Equation
\[eq:total\_r\] and $K$ in Equation \[eq:state\_value\] and
\[eq:action\_value\] are finite and the case where they are infinite. In
the former case, the *episode* is terminated after reaching a subset of
$S^+$ known as the *terminal states*. These states are denoted by $T$,
where $(T \cup S) \cap S^+$ and $T \cap S = \emptyset$. Such problems
are called to be *episodic*, and as can be imagined, a sequence from a
starting state to a terminal state is called an episode; in the case of
the latter, where an episode lasts till infinity, the problems is said
to be *continuous* [@rl-sutton; @2016-kaiser].

The definitions, or rather limitations of a finite horizon problem are
sufficient for the task that was tackled by this project, and as such,
no more will be said about this matter.

#### Model Free vs Model Based Methods

Figure \[fig:model\_based\_free\] succinctly illustrates the differences
between the paths taken by: (1) model free learning and (2) model based
learning. In model free learning, the agent does not attempt to build
knowledge of the environment, but rather uses the experience it gains
through iterative interactions with the environment to directly affect
its policy and value functions. In contrast, in model based methods, the
agent use these experiences to create a model of what it believes to be
the behaviour of its environment. It then uses the model to *simulate*
the behaviour of its environment to *plan* its policy and value function
[@rl-sutton; @2016-kaiser]. As stated by [@rl-sutton], *planning* here
is defined as any computational process that produces or improves a
policy when given a model of the environment. This model can be thought
of as being the *transition function* [@2016-kaiser; @2010-deisenroth],
the *transition probabilities* [@rl-sutton] or the *transition dynamics*
[@2010-deisenroth]. This can be notationally expressed as Equation
\[eq:transition\_function\].
$$\label{eq:transition_function} M_{ss^\prime}^a = P(s_{t+1} = s^\prime|s_t = s, a_t = a)$$
In [@rl-sutton], there is a clear distinction between *learning* and
*planning*; the former is where real experience is used via direct
interaction with the environment, while the latter is where simulated
experience is used via interaction with the model of the environment.
However, in the present work, due to a model based method being used,
the term *learning* will be used to mean *planning* as defined by
[@rl-sutton].

Specific literature that has been used to develop efficient methods of
creating this model of the environment will discussed in a later
section.

#### Value Function methods vs Policy Search methods

This classification describes the main strategies used by RL algorithms
to carry out its main function of finding the optimal policy. Due to its
importance, the next two sections provide the main strategy used by
each, and list and describe the most widely used, and some new
algorithms for each. Proceeding this, each will be compared with respect
to their application in robotic motion problems.\

Value Function Methods
----------------------
