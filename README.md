*This project has been created as part of the 42 curriculum by gchmilew.*

# Description

Le projet consist a la creation dun rag.
le rag(Retrieval-Augmented Generation) est une technique utiliser pour permetre un un llm de pouvoir puisser des connaissance dans une base de donnee (document perso fichier ou autre) afin de repondre au mieux au questio en ce basant sur c document.
le proccess du rag et de recuperer les document est de les couper en morceaux de max predifinie, tout en gardant au maximum le context. ce qui veut dire couper a la fin de chapitre pour le txt ou bien a la fin de class ou de metode pour le code ou en fonction des titre pour du markdown ... et de cree un json/base de donner
contenant tout les infos sur chaque decoupe. dou i vient les index de depart et de fin.

une fois cette decoupe faite on lenvoi a travers un algorythm  de calcul de score pour mot dans mon cas le BM25.
le BM25 attribuer (Index) un score en fonction de la repitition de la rareter des mots en fonction de la taille dun document tout en limitant le score pour des mot repeter trop souvent (ot de liason : a, is, it, for ...)

a partir de la on fait la meme chose pour un query et on recherche le score le plus proche dans lindexage et on liste les top reponse.

grace a tout cela on inject les donner des chunk pertinent au llm afin qui puisse avoir un context supplementaire que la question en elle meme et de repndre a partir de sont context.



# Instructions

le makefile dispo de ancement par default pour le projet

make install pour installer toute les dependance
make index pour lancer le chunkage suivi de l'indexage
make search pour avoir un score de reponse pour 1 seul query
make search_dataset pour lancer le document complet de query et avoir le score
make answer repond a une question via le llm en ce basant sur un search_dataset
make answer_dataset repond a toute les question dans le dataset
make evaluate permet ede calcul son score de son retriever sur les retreiver de reference.

# Resources

https://www.geeksforgeeks.org/python/python-how-to-make-a-terminal-progress-bar-using-tqdm/
https://www.w3schools.com/python/ref_module_ast.asp
https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/

https://github.com/feyninc/chonkie
https://docs.chonkie.ai/common/welcome
https://github.com/xhluca/bm25s



La sauvegarde native de bm25s est beaucoup plus rapide qu'un JSON pour trois raisons principales :

1. Binaire vs Texte
JSON est un format texte. Pour enregistrer un nombre comme 0.12345678, JSON écrit 10 caractères (10 octets).

En binaire (le format utilisé par bm25s via des fichiers .npy), ce même nombre est écrit sous forme de float32 qui prend toujours 4 octets fixes, peu importe sa précision.

2. Copie mémoire directe (Zero-Parsing)
En JSON : Au moment de charger, Python doit lire le texte caractère par caractère, analyser la syntaxe (virgules, crochets) et convertir chaque chaîne de caractères en nombre dans la RAM. C'est très gourmand en CPU.

En Binaire (bm25s) : Le programme fait une copie directe des octets du disque vers la mémoire (ou utilise du mmap). Le CPU n'a aucun calcul de conversion à faire : les données arrivent en RAM prêtes à être utilisées par les structures C/NumPy.

3. Compression des matrices creuses (Sparse Matrices)
Un index BM25 est une matrice géante remplie en majorité de zéros (la plupart des mots n'apparaissent pas dans la plupart des documents). bm25s stocke uniquement les valeurs non nulles et leurs positions sous forme de tableaux d'indices binaires (format CSR/CSC), ce qui réduit drastiquement la taille sur le disque.

# Additional

## System architecture

mon pipeline consiste a un decoupage couple a un indexage. apres je retrieve  les sources des reponse potentiel au question. et jenvoi tout ca dans le llm

## Chunking strategy

jutilise une librairie nommer chonkie https://docs.chonkie.ai/common/concepts.
elle permet offre la possibliter de faire des decoupe specifique en fonction du type de fichier.
elle possde une lsite de priorite de decoupe afin davoir un chunk qui tente datteindre la taille maximal tout en gardant un context logique. Pas de decoupe en plein milieu de phrase il va preferer couper apres un fonction si il a pas la palce de metre la suite.
ca lui arrive parfois de depasser legerement car il veut pas couper en plein milieu. donc je verifier les taille des chunk et jutilise un decoupeur brut ou je coupe a la limite de taille et attribue un overlap au chunk suivant afin de garder le context davant.

## Retrieval method
jutilise la librairy bm25s https://pypi.org/project/bm25s/0.1.5/
elle permet de faire l indexage et le retrieval.
elle gere la tokenization(calcul des score) ainsi que la comparaison entre le query et lindexage.

## Performance analysis

## Design decisions
## Challenges faced
## Example usage