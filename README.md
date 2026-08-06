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