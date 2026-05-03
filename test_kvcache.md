**Protocole de mesure de la latence de swap KV Cache sur SSD NVMe**

Ce document décrit la méthodologie à suivre pour valider expérimentalement la latence du mécanisme de pagination du cache KV (swap-out / swap-in) entre la VRAM et un SSD NVMe PCIe 4.0 dans un environnement SGLang. L’objectif est de déterminer si cette latence reste transparente pour un service d’inférence interactif.

---

## 1. Rappel du problème et enjeu

Le budget VRAM limité (24 Go, dont ~15 Go réservés au cache KV) impose de décharger le cache KV de certaines requêtes en attente sur le SSD lorsque la mémoire est saturée, notamment pour libérer de la place au profit de requêtes prioritaires (Fast Lane). Si le temps de déchargement / rechargement est trop élevé, la génération de tokens sera bloquée, entraînant des pauses perceptibles et une dégradation de l’expérience utilisateur. La cible maximale tolérée est de **20 ms par opération** ; au-delà, le système est considéré comme non viable en l’état.

## 2. Configuration matérielle et logicielle

- **GPU** : NVIDIA RTX 3090 ou 4090 (24 Go VRAM), pilotes ≥ 535, CUDA ≥ 12.1.
- **SSD** : NVMe PCIe 4.0 avec vitesse de lecture séquentielle ≥ 5 Go/s (ex. Samsung 980 Pro, WD Black SN850). Température contrôlée pour éviter le throttling.
- **RAM système** : 64 Go.
- **SGLang** : version ≥ 0.2.0, compilée avec support CUDA/Triton.
- **Modèle** : Qwen-2.5-14B-INT4 (ou équivalent 14-20B quantifié), occupant ~9 Go en VRAM.
- **Système d’exploitation** : Ubuntu 22.04, noyau 5.15+ avec IOMMU désactivé si pertinent.
- **Monitoring** : `nvidia-smi`, `iostat`, `fio`, scripts python avec `torch.cuda.Event`.

## 3. Métriques à mesurer

1. **Latence de swap-out** (écriture VRAM → SSD) : temps durant lequel le moteur d’inférence est bloqué pour libérer la mémoire.
2. **Latence de swap-in** (lecture SSD → VRAM) : temps nécessaire pour recharger un bloc de cache KV suspendu avant de reprendre la génération.
3. **Impact sur le flux token** :
   - **Time-to-first-token (TTFT)** supplémentaire pour une requête dont le cache a été swappé.
   - **Inter-token latency (ITL)** : écart entre deux tokens successifs lors d’une reprise après swap.
4. **Throughput** (tokens/seconde) avant, pendant et après l’opération de swap.
5. **Bande passante effective du SSD** dans les conditions du test (lecture/écriture aléatoire ou séquentielle selon la politique de pagination).

## 4. Scénarios de test

### Scénario A – Swap-out déclenché par saturation (requêtes longues)
1. Lancer 3 requêtes simultanées avec des contextes de 4096 tokens et une génération longue (1024 tokens).
2. Surveiller la VRAM ; lorsque `max_total_num_tokens` (18 000) est atteint, une 4ème requête Fast Lane est introduite.
3. Le système doit alors suspendre une requête Spot Lane, swapper son cache KV, et servir la Fast Lane.
4. Mesurer le temps de blocage de la Fast Lane due au swap-out (elle ne peut pas commencer tant que la place n’est pas libérée) et l’ITL de la requête suspendue après reprise.

### Scénario B – Swap-in à la reprise
1. Après un swap-out, terminer la requête prioritaire.
2. Dès que le budget le permet, le système recharge le cache de la requête suspendue et reprend sa génération.
3. Mesurer le temps avant la reprise effective du premier token après swap-in.

### Scénario C – Micro-benchmark de transfert brut
1. Utiliser `fio` pour mesurer la latence de lecture/écriture sur le SSD avec des blocs de taille caractéristique du cache KV (plusieurs centaines de Mo).
2. Comparer avec des mesures via un script PyTorch qui alloue un tenseur en VRAM, le copie en RAM via `cudaMemcpy`, puis l’écrit sur SSD, et inversement, en utilisant des events CUDA pour une précision à la microseconde.

### Scénario D – Charge P2P réaliste
1. Simuler une arrivée continue de requêtes Spot Lane (10 req/s) et des bursts de Fast Lane (toutes les 5 secondes).
2. Mesurer la latence de swap sous stress, avec également du transfert réseau (BitSwap) en arrière-plan pour observer les interférences.

## 5. Procédure de mesure détaillée

### 5.1. Préparation de l’environnement
- Désactiver les services inutiles, fixer la fréquence GPU (nvidia-smi -lgc) pour éviter le throttling.
- Utiliser un système de fichiers à faible overhead (ext4 en mode `noatime`).
- Vider le cache disque avant chaque mesure (`echo 3 > /proc/sys/vm/drop_caches`).

### 5.2. Instrumentation de SGLang
- Activer les logs détaillés : `--log-level debug` et rediriger vers un fichier.
- Ajouter des points de mesure avec `torch.cuda.Event` dans le code de pagination si l’on souhaite une précision fine. À défaut, analyser les timestamps des logs de SGLang qui indiquent le début et la fin d’un swap.

### 5.3. Exécution des scénarios
- Pour chaque scénario, effectuer 10 répétitions et enregistrer les métriques brutes.
- Pour le scénario A, enregistrer l’heure d’arrivée de la Fast Lane, l’heure de début effectif de sa génération après swap-out, et les temps inter-token.
- Pour le scénario B, enregistrer l’heure où la requête suspendue redevient active et l’écart avant le premier token.

### 5.4. Outils de capture
- **nvidia-smi** en boucle (toutes les 100 ms) pour suivre l’utilisation VRAM.
- **iostat -x 1** pour la latence disque (await, svctim).
- **dstat** ou **vmstat** pour la mémoire système et les E/S.
- Script Python lançant les requêtes et enregistrant les latences de bout en bout.

## 6. Analyse des résultats et seuils

- **Swap-out latency** : doit être < 10 ms pour ne pas impacter la Fast Lane. Une latence de 20-30 ms est limite mais peut être acceptable si elle reste rare.
- **Swap-in latency** : une latence jusqu’à 50 ms peut être acceptable pour une requête Spot Lane suspendue, car la génération reprend sans que l’utilisateur ait forcément une attente immédiate (il n’est plus en train de regarder activement). Toutefois, la reprise doit rester fluide.
- **Inter-token latency** après reprise : doit revenir à la normale (< 50 ms) immédiatement après le premier token.
- **Bande passante effective** : doit être > 3 Go/s pour des blocs de 100-200 Mo ; si elle est inférieure, le SSD ou le bus est mal configuré.

Si les mesures dépassent les seuils, envisager :
- Utilisation d’un SSD NVMe plus rapide (PCIe 5.0) ou d’un RAID 0 de SSD.
- Réduction du nombre de tokens swappés en segmentant le cache KV (ne swapper que les couches les plus anciennes).
- Passage à un modèle plus petit (10-12B) pour libérer plus de VRAM et éviter le swap.

## 7. Scripts et commandes utiles

**Boucle nvidia-smi :**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 0.1 > vram.log
```

**Benchmark SSD avec fio (lecture aléatoire, blocs de 128 Mo) :**
```bash
fio --name=readtest --rw=randread --bs=128M --size=1G --numjobs=1 --runtime=30 --time_based --ioengine=libaio --direct=1 --group_reporting
```

**Mesure de transfert CPU-GPU avec PyTorch (indicateur pour le swap) :**
```python
import torch
import time

# Allouer un tenseur en VRAM
tensor_gpu = torch.randn(1000, 1000, device='cuda')
torch.cuda.synchronize()

# Mesurer copie GPU -> CPU (swap-out)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
tensor_cpu = tensor_gpu.cpu()
end.record()
torch.cuda.synchronize()
print(f"Copy GPU->CPU time: {start.elapsed_time(end)} ms")
```

> Pour simuler l’écriture sur SSD, il suffit d’enregistrer le temps de la même copie en incluant un `torch.save`.

## 8. Pièges à éviter

- **Throttling thermique** : vérifier la température du SSD ; un SSD NVMe peut brider après quelques secondes d’utilisation intense. Utiliser un dissipateur thermique.
- **Cache du système de fichiers** : la première écriture peut être rapide (cache DRAM), les suivantes révèlent la vraie latence. Utiliser `direct=1` pour les mesures disque.
- **Fragmentation GPU** : après de nombreux swaps, l’allocateur GPU peut fragmenter la mémoire et augmenter les temps d’allocation. Nettoyer entre les tests et mesurer également le temps d’allocation.
- **Interférences réseau** : exécuter les tests de latence swap sans trafic BitSwap concurrent pour isoler la variable.

---

**Livrable attendu** : un rapport contenant les métriques brutes et agrégées (moyenne, p99, max) pour chaque scénario, ainsi qu’une conclusion sur la viabilité de la configuration matérielle cible. Si la latence dépasse 20 ms en p99, une révision du dimensionnement ou du mécanisme de swap sera nécessaire avant la phase 1 du MVP.