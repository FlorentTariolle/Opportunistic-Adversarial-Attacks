% v0.01 2023/11/27 v0.01 LaTeX Class for CAp 2024 submissions

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cap2026}[2023/11/27 v0.01 LaTeX Class for CAp 2026 submissions]

\RequirePackage{ifthen}
\newboolean{isfrench}
\newboolean{isnatbib}
\newboolean{islnum}

\DeclareOption{no-natbib}{%
    \setboolean{isnatbib}{false}%
}%
\DeclareOption{natbib}{%
    \setboolean{isnatbib}{true}%
}%

\DeclareOption{french}{%
    \setboolean{isfrench}{true}%
}%
\DeclareOption{english}{%
    \setboolean{isfrench}{false}%
}%

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{english}
\ExecuteOptions{natbib}
\ProcessOptions\relax
\LoadClass[a4paper,11pt]{article}

\RequirePackage[utf8]{inputenc}%
\RequirePackage{authblk}
\RequirePackage{hyperref}
\ifthenelse{\boolean{isnatbib}}{%
\RequirePackage{natbib}
}{%
}%
\RequirePackage[width=17cm,height=24cm]{geometry}
\ifthenelse{\boolean{isfrench}}{%
    \RequirePackage[T1]{fontenc}%
    \RequirePackage[french]{babel}%
    \renewcommand\Authands{, et }%
    \renewcommand\Authand{ et }%
}{%
}%

% Theorems
\RequirePackage{amsthm}
% Bold title in theorems
\makeatletter
\def\th@plain{%
  \thm@notefont{}% same as heading font
  \itshape % body font
}
\def\th@definition{%
  \thm@notefont{}% same as heading font
  \normalfont % body font
}
\makeatother

\ifthenelse{\boolean{isfrench}}{%
    \newtheorem{theorem}{Théorème}%
    \newtheorem{definition}[theorem]{Définition}%
    \newtheorem{lemma}[theorem]{Lemme}%
    \newtheorem{proposition}[theorem]{Proposition}%
    \newtheorem{corollary}[theorem]{Corollaire}%
}{%
    \newtheorem{theorem}{Theorem}%
    \newtheorem{definition}[theorem]{Definition}%
    \newtheorem{lemma}[theorem]{Lemma}%
    \newtheorem{proposition}[theorem]{Proposition}%
    \newtheorem{corollary}[theorem]{Corollary}%
}%

% Keywords
\ifthenelse{\boolean{isfrench}}{%
\newcommand{\keywords}[1]{\noindent\textbf{Mots-clefs}: #1.}
}{%
\newcommand{\keywords}[1]{\noindent\textbf{Keywords}: #1.}
}%

% Lines Numbering
\RequirePackage{lineno}
\RequirePackage{accsupp}
\renewcommand{\thelinenumber}{% Line number printing mechanism
  \BeginAccSupp{ActualText={}}\arabic{linenumber}\EndAccSupp{}%
}
\linenumbers