\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{fict}[2022/12/20 Faculty of ICT Dissertation template]

% Load the book class.
\LoadClass[12pt, a4paper, oneside, openany]{book}

% Use OTF fonts and set Lato as the default font.
\RequirePackage{fontspec, enumitem, adjustbox}
\RequirePackage[default]{lato}

% Set the margins of the document.
\RequirePackage[a4paper, top=2.54cm, bottom=2.54cm, left=3cm, right=2cm]{geometry}

% Set the text to be left-aligned and indent of 1.27cm
\raggedright{}
\setlength{\parindent}{1.27cm}

% Set 1.5 line spacing.
\RequirePackage{setspace}
\onehalfspacing{}

% Enable figure insertion.
\RequirePackage{graphicx}

% Generate hyperlinks.
\RequirePackage[hidelinks,pdfusetitle]{hyperref}

% Math symbols and access to the '\eqref' command.
\RequirePackage{amsmath, amssymb, amsfonts}% For Rel Numbers added final two 03/05/2025
\RequirePackage{booktabs, multirow} % For channels table 06/05/2025
\RequirePackage{changepage} % For COCO object size 08/05/2025

% For quotes
\RequirePackage{epigraph}

% Reference style and engine.
\RequirePackage[bibstyle=ieee,citestyle=numeric-comp]{biblatex}

% Table configuration.
\RequirePackage{tabularray}
%% For aligning numbers in a table with the period symbol.
\RequirePackage{siunitx}
\sisetup{detect-all}
\UseTblrLibrary{siunitx}

% Chapter page configuration.
\RequirePackage{titlesec}
\RequirePackage{apptools}
%% Change the chapter style.
\titleformat{\chapter}[hang]{\LARGE\bfseries}{\IfAppendix{\appendixname{}}{Chapter} \thechapter}{1em}{}% NEW Modified
\titlespacing*{\chapter}{0pt}{0pt}{12pt}

%% Remove the spacing between the title and the border.
\makeatletter
\def\ttl@mkchap@i#1#2#3#4#5#6#7{%
    \ttl@assign\@tempskipa#3\relax\beforetitleunit
    \vspace{\@tempskipa}%
    \global\@afterindenttrue
    \ifcase#5 \global\@afterindentfalse\fi
    \ttl@assign\@tempskipb#4\relax\aftertitleunit
    \ttl@topmode{\@tempskipb}{%
        \ttl@select{#6}{#1}{#2}{#7}}%
    \ttl@finmarks  % Outside the box!
    \@ifundefined{ttlp@#6}{}{\ttlp@write{#6}}}
\makeatother

% Table of contents configuration.
\RequirePackage{tocloft}
%% List of figures configuration.
\renewcommand{\cftfigfont}{Figure }
\renewcommand{\cftfigaftersnum}{}
\renewcommand{\cftfigindent}{0pt}
%% List of tables configuration.
\renewcommand{\cfttabaftersnum}{}
\renewcommand{\cfttabfont}{Table }
\renewcommand{\cfttabindent}{0pt}

% Caption configuration.
\RequirePackage[justification=centering]{caption}
\captionsetup{labelsep=space}

% Algorithm environment.
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

% Add color support.
\RequirePackage{xcolor}

% Add listings support.
\RequirePackage{listings}
%% Define custom listing style.
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{1,1,0.95}
\definecolor{codegreen2}{RGB}{41, 196, 102}
\definecolor{codegray2}{RGB}{190, 200, 198}
\definecolor{codeblue2}{RGB}{13, 162, 210}
\definecolor{codeyellow2}{RGB}{244, 169, 49}
\definecolor{backcolour2}{RGB}{242, 245, 249}
\definecolor{codeblack2}{RGB}{20, 36, 59}

% \lstdefinestyle{mystyle}{
%     backgroundcolor=\color{backcolour},
%     commentstyle=\color{codegreen},
%     keywordstyle=\color{magenta},
%     numberstyle=\tiny\color{codegray},
%     stringstyle=\color{codepurple},
%     basicstyle=\ttfamily\footnotesize,
%     breakatwhitespace=false,
%     breaklines=true,
%     captionpos=b,
%     keepspaces=true,
%     numbers=left,
%     numbersep=5pt,
%     showspaces=false,
%     showstringspaces=false,
%     showtabs=false,
%     tabsize=2
% }
\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour2},
    commentstyle=\color{codeblue2},
    keywordstyle=\color{codegreen2},
    numberstyle=\tiny\color{codegray2},
    stringstyle=\color{codeyellow2},
    basicstyle=\ttfamily\footnotesize\color{codeblack2},
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2
}
\lstset{style=mystyle}

% Add Corollary, theorems, and lemmas support
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{corollary}{Corollary}[theorem]
\newtheorem{lemma}[theorem]{Lemma}

% Glossary and acronym management.
\RequirePackage[automake,acronym]{glossaries}
%% Remove the bold font when listing abbreviations and glossary terms.
\renewcommand*{\glsnamefont}[1]{\textmd{#1}}
%% Command required to generate glossaries.
\makeglossaries{}

% Header & Footer configuration.
\RequirePackage{fancyhdr}

% For subsubsections
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

%% Override the chapter and section marks of the default 'fancy' style.
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{#1}}
\renewcommand{\subsubsectionmark}[1]{\markright{\thesubsubsection\ #1}}% For subsubsections

%% Create a style that only shows page numbers.
\fancypagestyle{pageNumbersOnly}
{
    \fancyhf{}
    \rhead{}
    \lhead{}
    \cfoot{\fancyplain{}{\thepage}}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

%% Update the Main style (Changed Chapter line).
% \fancypagestyle{MainMatter}
% {
%     \fancyhf{}
%     \lhead{}
%     \rhead{\fancyplain{}{\thechapter{} \leftmark}}
%     \cfoot{\fancyplain{}{\thepage}}
%     \renewcommand{\headrulewidth}{0pt}
%     \renewcommand{\footrulewidth}{0pt}
% }
\fancypagestyle{MainMatter}
{
    \fancyhf{}
    \lhead{}
    \rhead{\fancyplain{}{\IfAppendix{\appendixname{}}{Chapter} \thechapter{}. \leftmark \vspace{-7pt} \\ \rule{\textwidth}{0.7pt}}} % Adjusted spacing
    \cfoot{\fancyplain{}{\thepage}}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

%% Set the header to a fixed height of 15.8pt.
\setlength{\headheight}{15.8pt}

% Custom commands.
\makeatletter
\newcommand\supervisor[1]{\newcommand\@supervisor{#1}}
\newcommand\cosupervisor[1]{\newcommand\@cosupervisor{#1}}
\newcommand\titledate[1]{\newcommand\@titledate{#1}}
\newcommand\degreename[1]{\newcommand\@degreename{#1}}
\makeatother
