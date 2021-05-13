# Results


## Exp01: Number of CSP features
- normalised
- rereferenced
- lognorm

### 2 CSP Features


Fold |Accuracy	| True imagery_handL	| False imagery_handL	| False imagery_handR	| True imagery_handR |
--- |--- |--- |--- |--- |--- |
Fold 0 | 0.53		| 20		| 18		| 17		| 20 |
Fold 1 | 0.72		| 23		| 7		| 14		| 31 |
Fold 2 | 0.61		| 25		| 17		| 12		| 21 |
Fold 3 | 0.72		| 29		| 13		| 8		| 25 |
Fold 4 | 0.80		| 27		| 5		| 10		| 33 |
**Average** |**0.68**		| 24.8		| 12.0		| 12.2		| 26.0 |


### 4 CSP Features
Fold |Accuracy	| True imagery_handL	| False imagery_handL	| False imagery_handR	| True imagery_handR |
--- |--- |--- |--- |--- |--- |
Fold 0 | 0.55		| 26		| 23		| 11		| 15 |
Fold 1 | 0.75		| 25		| 7		| 12		| 31 |
Fold 2 | 0.64		| 23		| 13		| 14		| 25 |
Fold 3 | 0.60		| 25		| 18		| 12		| 20 |
Fold 4 | 0.76		| 25		| 6		| 12		| 32 |
**Average** |**0.66**		| 24.8		| 13.4		| 12.2		| 24.6 |


### 6 CSP Features
Fold |Accuracy	| True imagery_handL	| False imagery_handL	| False imagery_handR	| True imagery_handR |
--- |--- |--- |--- |--- |--- |
Fold 0 | 0.55		| 27		| 24		| 10		| 14 |
Fold 1 | 0.76		| 26		| 7		| 11		| 31 |
Fold 2 | 0.65		| 21		| 10		| 16		| 28 |
Fold 3 | 0.61		| 23		| 15		| 14		| 23 |
Fold 4 | 0.75		| 25		| 7		| 12		| 31 |
**Average** |**0.66**		| 24.4		| 12.6		| 12.6		| 25.4 |

### 8 CSP Features
Fold |Accuracy	| True imagery_handL	| False imagery_handL	| False imagery_handR	| True imagery_handR |
--- |--- |--- |--- |--- |--- |
Fold 0 | 0.55		| 34		| 31		| 3		| 7 |
Fold 1 | 0.75		| 26		| 8		| 11		| 30 |
Fold 2 | 0.67		| 24		| 12		| 13		| 26 |
Fold 3 | 0.64		| 24		| 14		| 13		| 24 |
Fold 4 | 0.71		| 20		| 5		| 17		| 33 |
**Average** |**0.66**		| 25.6		| 14.0		| 11.4		| 24.0 |

___
## Exp02: Without Normalisation
- rereferenced
- 2 CSP features
- lognorm

Fold |Accuracy	| True imagery_handL	| False imagery_handL	| False imagery_handR	| True imagery_handR |
--- |--- |--- |--- |--- |--- |
Fold 0 | 0.54		| 17		| 15		| 17		| 20 |
Fold 1 | 0.70		| 18		| 5		| 16		| 30 |
Fold 2 | 0.59		| 23		| 17		| 11		| 18 |
Fold 3 | 0.58		| 22		| 17		| 12		| 18 |
Fold 4 | 0.83		| 29		| 7		| 5		| 28 |
**Average** |**0.65**		| 21.8		| 12.2		| 12.2		| 22.8 |


___
## Exp03: Without Rereferencing
- normalised
- 2 CSP features
- lognorm

Fold |Accuracy	| True imagery_handL	| False imagery_handL	| False imagery_handR	| True imagery_handR |
--- |--- |--- |--- |--- |--- |
Fold 0 | 0.57		| 20		| 15		| 17		| 23 |
Fold 1 | 0.68		| 21		| 8		| 16		| 30 |
Fold 2 | 0.56		| 25		| 21		| 12		| 17 |
Fold 3 | 0.65		| 26		| 15		| 11		| 23 |
Fold 4 | 0.75		| 22		| 4		| 15		| 34 |
**Average** |**0.64**		| 22.8		| 12.6		| 14.2		| 25.4 |


---
## Exp04: Using Bandpower
- normalised
- rereferenced
- 2 CSP features

Fold |Accuracy	| True imagery_handL	| False imagery_handL	| False imagery_handR	| True imagery_handR |
--- |--- |--- |--- |--- |--- |
Fold 0 | 0.60		| 29		| 22		| 8		| 16 |
Fold 1 | 0.77		| 33		| 13		| 4		| 25 |
Fold 2 | 0.65		| 31		| 20		| 6		| 18 |
Fold 3 | 0.67		| 30		| 18		| 7		| 20 |
Fold 4 | 0.80		| 29		| 7		| 8		| 31 |
**Average** |**0.70**		| 30.4		| 16.0		| 6.6		| 22.0 |