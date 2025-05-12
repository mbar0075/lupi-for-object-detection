[[_TOC_]]

## Using the Template

The template is provided with all the necessary examples and requires little
modification from the end user in order to start working on it.

### Customising the Title Page

The title page is automatically generated based on the parameter values provided
by the user in the `main.tex` file.
The commands that need to be changed are the following
```latex
\title{Sample title}
\author{Name Surname}
\supervisor{Prof.\ Name Surname}
\cosupervisor{Dr Name Surname}
\degreename{Something}
\titledate{June 2025}
```
A brief explanation of each command is given below:

|Command      | Description                                                                               |
|-------------|-------------------------------------------------------------------------------------------|
|title        | The project's title.                                                                      |
|author       | The project's author.                                                                     |
|supervisor   | The project's main supervisor.                                                            |
|cosupervisor | The project's co-supervisor. This command should be removed if there is no co-supervisor. |
|degreename   | The name of the degree.                                                                   |
|titledate    | The date to be displayed in the title page.                                               |

### Directory Structure

All files are stored in the `content` directory.
The `content` directory contains files for the title page, glossary entries,
acronyms, the abstract and the acknowledgements section.
The order in which these files need to be included is already provided as part
of the `main.tex` file.
The chapters and appendices are stored in the folders with the same name.
Each chapter has its own folder, with a folder to store the figures used in that
chapter.
This is only a recommendation of how to organise the LaTeX source file.
The author is free to amend this folder structure as they deem fit.

**NOTE** that the figures `chapters/figures` folder and the `ict_logo.pdf` file
in it must remain unchanged as these are referenced by the `title.tex` file
which generates the title page.

## Building the Template

A `makefile` is provided to build the document using `latexmk`.
To build the document run the command
```bash
make
```
This will generate the `main.pdf` file.
All the `LaTeX` files used for document generation will be stored in the `build`
folder.
To `clean` the main directory from the `build` folder, and any generated `pdf`
documents, the following command can be used
```bash
make clean
```

### LaTeX Compiler

This template requires the use of the `LuaTeX` compiler due to the use of the
`fontspec` package for out of the box UTF-8 font support.

### Embed fonts

It is recommended that when sharing a `pdf` document, all the fonts using in the
document are embedded in the said `pdf` document in the instance where the
recipient does not have any of the fonts used installed.
The provided `makefile` has a command that does this.
To do this, run the below command
```
make font_embed
```

## Overleaf support

This template has been tested and confirmed to work on Overleaf.

## Limitations

Note that the `makefile` has been developed and tested to be used on the Linux
Operating System.
MacOS users should have no problem in using the same `makefile`; however, this
has not been tested.
Windows users can either opt to compile the document on Linux through the
`Windows Subsystem for Linux (WSL)`, which is confirmed to work or through the
LaTeX editor's compile command.
If you will be using the latter, do not forget to set `LuaTeX` as the compiler.
