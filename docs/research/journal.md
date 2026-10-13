# Journal de recherche

Règle du journal : l'hypothèse et ses critères de rejet sont écrits **avant** le premier run.
Les résultats sont rapportés tels quels, échecs compris. Modifier la grille après avoir vu le
résultat ouvre une nouvelle hypothèse, jamais une retouche de l'ancienne.

---

## H1 — Dual momentum sur ETF (2026-10-13)

**Hypothèse économique.** Les rendements passés sur 6 à 12 mois persistent quelques mois
(momentum cross-sectionnel, Jegadeesh & Titman 1993 ; momentum temporel, Moskowitz, Ooi &
Pedersen 2012). Explication usuelle : sous-réaction des investisseurs et lenteur des flux.
Le filtre de tendance absolue (Antonacci 2014) évite de détenir un actif en tendance baissière,
source des pires drawdowns du momentum relatif.

**Protocole fixé avant le test.**
- Univers `etf_core` (14 ETF, sans biais de survie), 2007-06 → 2026-10, rebalancement mensuel,
  signal à la clôture t, exécution à la clôture t+1.
- Frais : 0,005 $/action, minimum 1 $, slippage 5 pb (approximation IBKR).
- Paramètres de base : lookback 252, skip 21, top 3, filtre absolu, refuge IEF.
- Grille : lookback {126, 189, 252} × skip {0, 21} × top_n {1, 2, 3} = 18 combinaisons.
- Walk-forward : 5 ans d'entraînement, 1 an de test, critère Sharpe.
- **Critères de rejet** : Sharpe déflaté < 0,90 ; Sharpe walk-forward hors échantillon < 0,30 ;
  moins de 50 % de la grille avec un Sharpe supérieur au benchmark équipondéré.

**Résultats (run `95241a3e77f3`, rapport `reports/output/95241a3e77f3.html`).**

| | Sharpe | CAGR | Vol | Max DD | Frais/an |
|---|---|---|---|---|---|
| Dual momentum (252/21/3), in-sample | 0,70 | 9,3 % | 14,2 % | −28,8 % | 32 pb |
| SPY buy & hold | 0,62 | 10,8 % | 19,7 % | −55,2 % | ~0 |
| Équipondéré mensuel (14 ETF) | 0,65 | 6,5 % | 10,5 % | −32,2 % | faible |
| Dual momentum, capital réel 3 000 $ | 0,67 | 8,6 % | | | 74 pb |

- Sharpe déflaté : **0,965** pour 271 essais comptabilisés (passe le seuil 0,90).
- Bootstrap par blocs, IC 95 % du Sharpe : [0,29 ; 1,13].
- Walk-forward hors échantillon (2013 → 2026-10) : Sharpe **0,70**, CAGR 11,0 %, max DD −30,8 %
  (passe le seuil 0,30). Sur la même période : SPY Sharpe 0,92, CAGR 15,1 %, max DD −33,7 % ;
  équipondéré Sharpe 0,78, max DD −19,2 %.
- Grille : Sharpe min 0,33, médian **0,49**, max 0,70. Seules **2 combinaisons sur 18 (11 %)**
  battent le Sharpe de l'équipondéré. **Échoue le critère de robustesse (50 %).**
- Les paramètres « de manuel » (252/21/3) sont précisément le meilleur point de la grille.
  C'est le signal classique d'un paramétrage sélectionné sur le passé par la littérature elle-même.
- Le walk-forward a changé de paramètres 6 fois en 14 ans ; les années 2015, 2016, 2018 sont
  négatives (whipsaw : le filtre sort puis rentre après des baisses courtes).

**Conclusion : H1 rejetée en l'état.** Le dual momentum sur cet univers réduit bien les
drawdowns par rapport au S&P 500 (−29 % contre −55 % en 2008), ce qui est sa promesse, mais il
ne bat ni le S&P ni l'équipondéré en Sharpe hors échantillon sur 2013-2026, et son avantage
in-sample tient à un seul point de la grille. Avec 3 000 $, les frais doublent (74 pb/an) à cause
du minimum de commission sur une rotation de 5,7× par an.

**Ce que ça apprend.**
1. Le pipeline fonctionne de bout en bout et le protocole a fait son travail : sans la grille et
   les critères écrits à l'avance, le Sharpe 0,70 > 0,62 aurait été présenté comme un succès.
2. La période hors échantillon est un marché haussier quasi continu, le pire régime pour une
   stratégie de rotation défensive. Ce n'est pas une excuse, c'est une information : la valeur
   du dual momentum est dans la protection, pas dans le rendement, et il faut la mesurer comme
   telle (ex. : critère Calmar ou drawdown, pas Sharpe).
3. Pistes pour H2 (à formuler séparément) : score de momentum ajusté de la volatilité ; cible de
   volatilité au niveau du portefeuille ; rebalancement avec bande de tolérance pour réduire le
   turnover ; comparaison au 60/40 plutôt qu'au SPY.

---
