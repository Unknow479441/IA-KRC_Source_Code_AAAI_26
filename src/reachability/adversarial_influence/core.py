\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2026}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{enumitem}
\frenchspacing

\pdfinfo{
/TemplateVersion (2026.1)
}

\begin{document}

\onecolumn

\appendix
\section{Appendix}
This appendix provides detailed specifications for the key components of the Interference-Aware K-Step Reachable Communication (IA-KRC) framework, including its core algorithm, the neural network for interference prediction, and the hyperparameters used in the experiments.

\subsection{A.1 Implementation Details of the IA-KRC Algorithm}
\subsubsection{A.1.1 Key Variables and Functions} Before presenting the algorithm, we define the key functions and variables used:

\textbf{Variables:}
\begin{itemize}
\item $d_{IA}(s_1, s_2)$: Interference-aware shortest transition distance (Definition 2)
\item $\mathcal{S}_{IA}(s_1, k)$: Interference-aware $K$-step reachable region (Definition 3)
\item $D_i$: Centrality score for agent $i$ (sum of distances to all other agents)
\item $\mathcal{L}$: Set of elected leaders
\item $\mathcal{F}$: Set of followers (non-leader agents)
\end{itemize}

\textbf{Functions:}
\begin{itemize}
\item $\texttt{update\_from\_sight}()$: Updates geometry layer with visual observations
\item $\texttt{interference\_prediction\_module}()$: Processes interference prediction and threat assessment
\item $\texttt{aggregate\_graph}()$: Combines all layer edges using minimum weights
\item $\texttt{dijkstra\_reachable}()$: Modified Dijkstra with cost budget constraint
\end{itemize}

\begin{algorithm}[h!]
\caption{IA-KRC Multi-Layer Grouping Algorithm}
\label{alg:ia-krc}
\begin{algorithmic}[1]
\STATE \textbf{Input:} Agent states $\{s_i\}_{i=1}^N$, enemy observations $E_{obs}$, environment observations $env_{obs}$, parameters $K, N_L$
\STATE \textbf{Output:} Collaborative groups $G = \{G_1, G_2, ..., G_{N_L}\}$

\STATE \COMMENT{Phase 1: Multi-Layer Map Update}
\STATE Initialize layers: $L_g$, $L_t$, $L_c$
\STATE Extract $\texttt{agent\_pos}$, $\texttt{map\_matrix}$ from observations
\STATE $L_g.$\texttt{update\_from\_sight}($\texttt{agent\_pos}$, $\texttt{map\_matrix}$, $\texttt{sight\_range}$)
\STATE $L_t.$\texttt{update}($\texttt{agent\_transitions}$)
\STATE $L_c.$\texttt{update}($\texttt{experience\_stats}$, $L_t$)

\STATE \COMMENT{Phase 2: Interference Prediction}
\IF{interference enabled}
    \STATE $\texttt{interference\_costs} \leftarrow$ \texttt{interference\_prediction\_module}($E_{obs}$, $\texttt{agent\_pos}$)
    \STATE Integrate costs into $L_g$
\ENDIF

\STATE \COMMENT{Phase 3: Graph Aggregation and Reachability}
\STATE $\mathcal{G}_{agg} \leftarrow$ \texttt{aggregate\_graph}($L_g$, $L_t$, $L_c$)
\FOR{each agent $i$}
    \STATE $\mathcal{S}_{IA}(s_i, K) \leftarrow$ \texttt{dijkstra\_reachable}($\mathcal{G}_{agg}$, $s_i$, $K$)
    \STATE $D_i \leftarrow \sum_{j \neq i} d_{IA}(s_i, s_j)$
\ENDFOR

\STATE \COMMENT{Phase 4: Leader Election and Follower Assignment}
\STATE $\mathcal{L} \leftarrow$ top $N_L$ agents with lowest $D_i$ scores
\STATE Initialize groups: $G_l \leftarrow \{l\}$ for each $l \in \mathcal{L}$
\STATE $\mathcal{F} \leftarrow \{1, ..., N\} \setminus \mathcal{L}$

\FOR{each follower $f \in \mathcal{F}$}
    \STATE $\mathcal{L}_{cand} \leftarrow \{l \in \mathcal{L} : s_f \in \mathcal{S}_{IA}(s_l, K)\}$
    \IF{$\mathcal{L}_{cand} \neq \emptyset$}
        \STATE $l^* \leftarrow \arg\min_{l \in \mathcal{L}_{cand}} |G_l|$
        \STATE Add $f$ to group $G_{l^*}$
    \ENDIF
\ENDFOR

\STATE \textbf{return} $G = \{G_1, G_2, ..., G_{N_L}\}$
\end{algorithmic}
\end{algorithm}

\subsection{A.2 Interference Prediction Module}
The Interference Prediction Module quantifies the traversal risk by modeling the influence of adversarial agents as a potential field. In this field, high-threat enemies generate high-cost regions. The module's core is the calculation of a potential field for each enemy, which incorporates directional influence via a predicted attack intent angle $\theta$. This influence is modulated by a threat level, which is heuristically determined from the enemy's current state and recent actions. As detailed in Algorithm 2, a neural network predicts an attack intent vector for each enemy to derive the angle $\theta$. The resulting path cost map informs the IA-KRC framework's reachability calculations and agent coordination. Figure~\ref{fig:cost_map} shows an example of the dynamically computed transition cost map.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\linewidth]{AnonymousSubmission/LaTeX/figure3.png}
    \caption{Transition cost map computed from the start agent (yellow) after multi-layer map and interference map processing. }
    \label{fig:cost_map}
\end{figure}

\subsubsection{A.2.1 Key Variables and Functions for Algorithm 2}

\textbf{Variables:}
\begin{itemize}
\item $E_{obs}$: Visual observations of enemies.
\item $M_{influence}$: A grid representing the cumulative enemy influence across the map.
\item $M_{cost}$: A grid representing the pathfinding cost for each location.
\item $e$: An individual enemy entity.
\item $I_{\text{base}}$: Dynamically computed base influence strength for an enemy.
\item $\lambda_{\text{base}}$: Influence decay rate (hyperparameter).
\item $\alpha$: Angle influence factor (hyperparameter).
\item $\theta_e$: Predicted attack intent angle for enemy $e$.
\item $d_{\text{eff}}(p_1, p_2, \theta)$: Effective distance considering angle of attack.
\item $d_{\text{actual}}(p_1, p_2)$: Euclidean distance between two points.
\item $cost\_multiplier$: A factor to scale influence into path cost (hyperparameter).
\end{itemize}

\textbf{Functions:}
\begin{itemize}
\item $\texttt{extract\_enemies}(E_{obs})$: Parses observations to get a list of enemy entities and their states.
\item $\texttt{calculate\_influence}(e)$: Computes the dynamic base influence strength $I_{\text{base}}$ of an enemy based on its attributes (e.g., health, recent actions).
\item $\texttt{predict\_attack\_intent}(e)$: Predicts the attack intent angle $\theta_e$ for an enemy using a neural network.
\item $\texttt{normalize\_map}(M)$: Normalizes map values to a standard range.
\end{itemize}

\begin{algorithm}[h!]
\caption{Interference Prediction and Cost Calculation}
\label{alg:interference-prediction}
\begin{algorithmic}[1]
\STATE \textbf{Input:} Enemy observations $E_{obs}$, hyperparameters $\alpha, \lambda_{\text{base}}$
\STATE \textbf{Output:} Path cost map $M_{cost}$
\STATE \COMMENT{Phase 1: Initialize Maps}
\STATE Initialize $M_{influence}$ with zeros.
\STATE Initialize $M_{cost}$ with ones.
\STATE \COMMENT{Phase 2: Calculate Enemy Influence}
\STATE $Enemies \leftarrow \texttt{extract\_enemies}(E_{obs})$
\FOR{each enemy $e$ in $Enemies$}
    \STATE $I_{\text{base}} \leftarrow \texttt{calculate\_influence}(e)$ \COMMENT{Compute influence from state}
    \STATE $\theta_e \leftarrow \texttt{predict\_attack\_intent}(e)$ \COMMENT{Predict directional intent}
    \FOR{each cell $p$ on the map within influence range of $e$}
        \STATE $d_{\text{actual}} \leftarrow d_{\text{actual}}(p, e.position)$
        \STATE $d_{\text{eff}} \leftarrow d_{\text{actual}} \times (1 + \alpha(1 - \cos(\theta_e)))$ \COMMENT{Calculate effective distance}
        \STATE $I(p|e) \leftarrow I_{\text{base}} \times \exp(-\lambda_{\text{base}} \times d_{\text{eff}})$
        \STATE $M_{influence}[p] \leftarrow M_{influence}[p] + I(p|e)$
    \ENDFOR
\ENDFOR
\STATE \COMMENT{Phase 3: Compute Path Cost Map}
\STATE $M_{cost} \leftarrow 1.0 + cost\_multiplier \times M_{influence}$
\STATE For each obstacle position $p_{obs}$: $M_{cost}[p_{obs}] \leftarrow \infty$
\STATE $M_{cost} \leftarrow \texttt{normalize\_map}(M_{cost})$
\STATE \textbf{return} $M_{cost}$
\end{algorithmic}
\end{algorithm}

\subsubsection{A.2.2 Neural Network and Parameter Details}
This section provides further implementation details for the key components of the Interference Prediction Module.

\paragraph{Neural Network for Attack Intent Prediction.}
The attack intent angle $\theta_e$ for each enemy is derived from a predicted attack vector, which is generated by a dedicated neural network. This network takes an enemy's state as input and outputs a 2D vector representing its most likely direction of attack. The network architecture is a feed-forward Multi-Layer Perceptron (MLP) with two hidden layers:
\begin{itemize}[noitemsep]
    \item \textbf{Input Layer:} A flattened vector representing the enemy's state, including its recent trajectory (last 10 positions) and current health.
    \item \textbf{Hidden Layers:} The first hidden layer consists of 128 neurons with a ReLU activation function, followed by a second hidden layer of 64 neurons, also with ReLU activation.
    \item \textbf{Output Layer:} A linear layer with 2 neurons, producing the (x, y) components of the predicted attack intent vector.
\end{itemize}

\paragraph{Training Process.}
The attack intent prediction network is trained via supervised learning on data collected during simulation. For each enemy at each time step, we create a training sample consisting of its current state (the network input) and its actual movement vector over the next time step (the ground truth). The network is trained to minimize the angular difference between its predicted attack vector $\mathbf{v}_{\text{pred}}$ and the ground truth movement vector $\mathbf{v}_{\text{true}}$. The loss function is defined as the negative cosine similarity:
\[ \mathcal{L}_{\text{intent}} = 1 - \frac{\mathbf{v}_{\text{pred}} \cdot \mathbf{v}_{\text{true}}}{\|\mathbf{v}_{\text{pred}}\| \|\mathbf{v}_{\text{true}}\|} \]
Training is performed using the Adam optimizer with a learning rate consistent with the main policy training (see Table~\ref{tab:hyperparameters}).

\paragraph{Dynamic Influence Calculation.}
As stated in the main paper, the base influence strength $I_{\text{base}}$ for each enemy is computed dynamically based on real-time sampling of its state. The \texttt{calculate\_influence}(e) function in Algorithm 2 implements this process by deriving a heuristic score from the enemy's normalized health, recent movement patterns (e.g., speed and distance), and attack frequency. This dynamically computed value is then used directly in the potential field calculation.

\paragraph{Effective Distance ($d_{\text{eff}}$).}
The effective distance $d_{\text{eff}}$ is calculated using the formula $d_{\text{eff}} = d_{\text{actual}} (1 + \alpha (1 - \cos(\theta_e)))$. Here, $\theta_e$ is the angle between the neural network's predicted attack intent vector for enemy $e$ and the vector pointing from the enemy's position to the location $p$ being evaluated. This formulation ensures that the interference is strongest in the predicted direction of attack and weaker at other angles, creating a forward-facing cone of influence. The hyperparameter $\alpha$ controls the strength of this directional effect.

\subsection{A.3 Value Decomposition Framework Implementation Details}

The value decomposition framework in IA-KRC is built upon the centralized training with decentralized execution (CTDE) paradigm, extending the principles of value-based methods like QMIX to accommodate the dynamic grouping strategy. For each cooperative group $g$ formed by the IA-KRC mechanism, the framework learns a group-specific joint action-value function, $Q_{\text{tot}}^g$, which is trained to approximate the global team reward.

\paragraph{Group-Specific Value Function.} The core of our framework is the monotonic decomposition of the group's joint action-value function. $Q_{\text{tot}}^g$ is represented as a monotonic combination of the individual utility functions $Q_i$ for all agents $i \in g$. This property is crucial for decentralized execution, as it guarantees that a greedy action selection by each agent based on its local $Q_i$ corresponds to the maximization of the joint $Q_{\text{tot}}^g$. The relationship is formalized as $Q_{\text{tot}}^g = f_g(\{Q_i\}_{i \in g}, s)$, where $f_g$ is a mixing network specific to group $g$ and conditioned on the global state $s$. This architecture implies that each group effectively learns its own cooperative policy, tailored to its members and the current state, while still contributing to the global team objective.

\paragraph{Agent Architecture.} Each agent utilizes a recurrent neural network (RNN) with an attention mechanism, as indicated by the \texttt{comm\_imagine\_entity\_attend\_rnn} agent type in our configuration. This network processes the agent's local observation history $\tau_i$ to maintain a hidden state $h_i^t$. At each timestep, the RNN's input includes the agent's current observation, its previous action, and any messages received from its group leader. If the agent is itself a leader, it can also process messages from other leaders, a feature enabled by the \texttt{use\_leader\_comm} setting. The resulting hidden state $h_i^t$ is then fed into a feed-forward layer to compute the per-action Q-values $Q_i(\tau_i, \cdot)$.

\paragraph{Mixing Network.} The framework employs a mixing network architecture, specified as \texttt{flex\_qmix} in the configuration, to combine individual $Q_i$ values into the joint $Q_{\text{tot}}^g$. Consistent with QMIX, the weights of this mixing network are generated by a hypernetwork that takes the global state $s$ as input, allowing the mixing function to adapt to different environmental conditions. To enhance training stability, the mixing weights are normalized using a softmax function (\texttt{softmax\_mixing\_weights: True}).

\paragraph{Training Process.} The system is trained end-to-end by minimizing the total TD loss, summed across all dynamically formed groups, as defined in the main paper. Experiences are collected using parallel runners and stored in a replay buffer. The learner (\texttt{msg\_q\_learner}) samples mini-batches to perform updates. The TD target for each group $g$ is computed as:
\[ y_g^{\text{tot}} = r + \gamma \max_{\mathbf{a}'_g} Q_{\text{tot}}^g(\boldsymbol{\tau}'_g, \mathbf{a}'_g; \theta^-) \]
where $\theta$ and $\theta^-$ are the parameters of the online and target networks, respectively. The target network is periodically updated with the online network's parameters every 200 episodes (\texttt{target\_update\_interval: 200}).

\subsection{A.4 Hyperparameter Table}
\begin{table}[htbp]
    \centering
    \caption{Key Hyperparameter Settings}
    \label{tab:hyperparameters}
    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}} l r l r l r}
        \toprule
        \textbf{Parameter} & \textbf{Value} & \textbf{Parameter} & \textbf{Value} & \textbf{Parameter} & \textbf{Value} \\
        \midrule
        \multicolumn{6}{c}{\textbf{General RL Parameters}} \\
        \midrule
        Learning Rate & 5e-4 & Optimizer & Adam & Discount Factor ($\gamma$) & 0.99 \\
        Batch Size & 32 & Replay Buffer Size & 5000 & Target Update Interval & 200 ep. \\
        Epsilon Start & 1.0 & Epsilon Finish & 0.05 & Epsilon Anneal Time & 500k steps \\
        \midrule
        \multicolumn{6}{c}{\textbf{IA-KRC Framework and Interference Prediction}} \\
        \midrule
        K-Step Horizon ($K$) & 9 & Number of Leaders ($N_L$) & 3 & Cost Multiplier & 1.5 \\
        Interference Decay ($\lambda_{\text{base}}$) & 0.3 & Angle Influence ($\alpha$) & 0.5 & & \\
        Intent Net Hidden Dim 1 & 128 & Intent Net Hidden Dim 2 & 64 & & \\
        \midrule
        \multicolumn{6}{c}{\textbf{Agent and Mixer Architecture}} \\
        \midrule
        RNN Hidden Dim & 64 & Attention Dim & 128 & Attention Heads & 4 \\
        Mixing Net Dim & 32 & Hypernet Dim & 128 & & \\
        \bottomrule
    \end{tabular*}
\end{table}

\end{document} 