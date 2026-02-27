\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,graphicx}
\title{$title}
\author{$authors}
\date{\today}
\begin{document}
\maketitle

\section{Run Metadata}
Run ID: \texttt{$run_id}

\section{Model and Derivation}
Theory family: $derivation_family

\subsection*{Key Equations}
$equations

\section{Numerical Summary}
$qnm_summary

\section{Novelty Assessment}
Novelty score: $novelty_score

$novelty_explanation

\subsection*{Nearest Neighbors}
$neighbors

\bibliographystyle{unsrt}
\bibliography{refs}
\end{document}
