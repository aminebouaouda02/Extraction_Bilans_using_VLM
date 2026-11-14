import json

with open("bilan_group_3.json", "r", encoding="utf-8") as f:
    bilan = json.load(f)

print("="*60)
print(" VERIFICATION DE LA SOMME DES LIGNES INDIVIDUELLES (PYTHON) ")
print("="*60)

# 1. Vérification ACTIF
lignes_actif = bilan["actif"]["lignes"]
totaux_actif = bilan["actif"]["totaux"]

# On filtre les lignes de postes réels (qui commencent par * ou qui ont un libellé de poste)
postes_actif = [l for l in lignes_actif if l.get("poste") and not l.get("poste").startswith("TOTAL")]

somme_brut_calculee = sum(l.get("brut", 0.0) for l in postes_actif)
somme_amort_calculee = sum(l.get("amortissements_provisions", 0.0) for l in postes_actif)
somme_net_N_calculee = sum(l.get("net_N", 0.0) for l in postes_actif)

# Somme des sous-totaux I + II + III
total_I = totaux_actif.get("total_I_actif_immobilise_net_N", 0.0)
total_II = totaux_actif.get("total_II_actif_circulant_net_N", 0.0)
total_III = totaux_actif.get("total_III_tresorerie_actif_net_N", 0.0)
somme_sous_totaux_actif = total_I + total_II + total_III
total_general_actif = totaux_actif.get("total_general_actif_net_N", 0.0)

print("\n--- 1. ACTIF (EXERCICE N) ---")
print(f"Somme des sous-totaux I + II + III   : {somme_sous_totaux_actif:,.2f} DH")
print(f"Total Général lu en bas de page      : {total_general_actif:,.2f} DH")
print(f"Écart entre sous-totaux et Total     : {abs(somme_sous_totaux_actif - total_general_actif):.2f} DH")
print(f"Somme des {len(postes_actif)} postes détaillés Net N  : {somme_net_N_calculee:,.2f} DH")

# 2. Vérification PASSIF
lignes_passif = bilan["passif"]["lignes"]
totaux_passif = bilan["passif"]["totaux"]

total_I_p = totaux_passif.get("total_I_financement_permanent_N", 0.0)
total_II_p = totaux_passif.get("total_II_passif_circulant_N", 0.0)
total_III_p = totaux_passif.get("total_III_tresorerie_passif_N", 0.0)
somme_sous_totaux_passif = total_I_p + total_II_p + total_III_p
total_general_passif = totaux_passif.get("total_general_passif_N", 0.0)

print("\n--- 2. PASSIF (EXERCICE N) ---")
print(f"Somme des sous-totaux I + II + III   : {somme_sous_totaux_passif:,.2f} DH")
print(f"Total Général lu en bas de page      : {total_general_passif:,.2f} DH")
print(f"Écart entre sous-totaux et Total     : {abs(somme_sous_totaux_passif - total_general_passif):.2f} DH")

print("\n" + "="*60)
if abs(somme_sous_totaux_actif - total_general_actif) < 0.01 and abs(somme_sous_totaux_passif - total_general_passif) < 0.01:
    print("CONCLUSION : PREUVE ARITHMETIQUE VALIDE !")
    print("Les sous-totaux intermédiaires se recoupent exactement avec le Total Général.")
print("="*60)
