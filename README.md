# Simulateur de combinaisons de medicaments

## Exemple de donnees produites
| Rx1 | Rx2 | Rx3 | Rx4 | ... | RxN | RR   |
|-----|-----|-----|-----|-----|-----|------|
| 1   | 0   | 0   | 1   | ... | 0   | 3.00 |
| 0   | 1   | 0   | 1   | ... | 1   | 2.67 |
| 0   | 1   | 1   | 1   | ... | 1   | 3.14 |
| ⋮   | ⋮   | ⋮   | ⋮   | ... | ⋮   | ⋮    |
| 1   | 1   | 1   | 1   | ... | 0  | 1.85|


## Abbreviations
* RR = Risque relatif

## Configuration
* `file_identifier`: Base textuelle pour identifier le jeu de donnees
* `output_dir`: Dossier de sortie pour les fichiers du jeu de donnees
* `seed`: Graine aleatoire pour le generateur
* `n_combi`: Nombre de combinaisons a produire
* `n_rx`: Nombre de medicaments possibles (nombre de colonnes)
* `average_rx`: Nombre de medicaments moyen par combinaison (densite)

* `patterns`: Sous-configuration relative aux patrons dangereux
    * `n_patterns`: Nombre de patron dangereux a produire
    * `min_rr`: RR minimal des patrons
    * `max_rr`: RR maximal des patrons

* `disjoint_combinations`: Sous-configuration relative aux combinaisons disjointes des patrons dangereux
    * `mean_rr`: Moyenne de la normale pour le tirage du RR
    * `std_rr`: Variance de la normale pour le tirage du RR


* `inter_combinations`: Sous-configuration relative aux combinaisons ayant une intersection non vide avec des patrons dangereux
    * `std_rr`: Variance de la normale pour le tirage du RR
    * `similarity_std`: Booleen determinant si on utilise la similarite entre les patrons et les combinaisons afin de calculer le RR




## Distributions utilisees
### Patrons
Ici, on utilise des distributions uniformes dans l'intervalle [`patterns:min_rr`, `patterns:max_rr`] afin de faciliter la creation de jeux de donnees plus ou moins difficiles

### Combinaisons avec intersection avec un patron
On utilise une normale d'ecart-type`inter_combinations:std_rr` et avec une moyenne calculee a l'aide de la similarite entre les combinaisons et les patrons dangereux.

### Combinaisons disjointe des patrons
On utilise une normale de moyenne `disjoint_combinations:mean_rr` et d'ecart-type `disjoint_combinations:std_rr`. Les combinaisons reliees a un patron seront donc plus proche d'un RR predetermine par la configuration.