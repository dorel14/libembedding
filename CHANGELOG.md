# CHANGELOG


## v1.1.0 (2026-08-28)

### Chores

- **release**: Bump version 1.0.3 pour nouvelle release PyPI
  ([`73d9607`](https://github.com/dorel14/libembedding/commit/73d960717e49b844da6332ac5eb6048f6f6c66a6))

- **release**: Test semantic-release config with gitpython pin
  ([`45b4759`](https://github.com/dorel14/libembedding/commit/45b4759a98c8fbf70edd20f846cd5b0a5a6e96b7))

### Features

- **python**: Ajouter le pool d'embeddings multi-workers
  ([`28dc639`](https://github.com/dorel14/libembedding/commit/28dc639f45a10341062d758ae071bfca18c53eb4))

Ajout de la classe `TextEmbeddingPool` dans l'API Python, permettant la parallélisation au niveau
  des requêtes via plusieurs sessions ONNX distinctes. Cette approche est plus efficace que le
  parallélisme intra-op d'ORT pour les petits transformeurs, offrant jusqu'à 4x le débit d'une
  session unique.

Les composants associés ont également été ajoutés : - En-têtes C++ pour l'autotuneur, le sélecteur
  de modèle et le pool d'embeddings (`autotuner.h`, `embedding_pool.hpp`, `model_selector.h`) -
  Suite de benchmarks complète (CMakeLists.txt + 40+ fichiers) - Section "Performance
  Characteristics" dans le README documentant les résultats d'optimisation CPU - Correction de la
  compatibilité Windows pour le répertoire de cache (`USERPROFILE` avant `HOME`) - Fichier
  `CHANGELOG.md` et mise à jour de `.gitignore` (`.venv/`)


## v1.0.2 (2026-08-26)

### Continuous Integration

- Bump version 1.0.2 pour nouvelle release PyPI
  ([`117bab5`](https://github.com/dorel14/libembedding/commit/117bab5b2748b97155adea17a7e12adc6b35c7bb))


## v1.0.1 (2026-08-26)

### Bug Fixes

- Auditwheel step ne doit pas corrompre le nom de la wheel
  ([`dcfb8f9`](https://github.com/dorel14/libembedding/commit/dcfb8f907aa97ccfd18fe1dfc3be092efea34229))

- Enlever le pin pydantic v1, laisser semantic-release gerer ses deps
  ([`377402c`](https://github.com/dorel14/libembedding/commit/377402c540099858e06327c4b85effe7015764dc))

- Installer la wheel reparee aussi sur Windows (win_amd64)
  ([`8d63935`](https://github.com/dorel14/libembedding/commit/8d6393582b3edc35484117d7eade5b1563b5ff57))

- Installer la wheel reparee sur Linux (manylinux) et macOS (macosx)
  ([`76e1d33`](https://github.com/dorel14/libembedding/commit/76e1d339d3556f9124f6f90e842871bd49033d35))

- Installer seulement la wheel reparee (manylinux) sur Linux
  ([`017e820`](https://github.com/dorel14/libembedding/commit/017e820caf0bb15693b63265efb8730ea1a33b6a))

- List wheel contents avec plusieurs wheels apres auditwheel
  ([`8ad88da`](https://github.com/dorel14/libembedding/commit/8ad88da53d6f42e05a9ada1dcb5b14372b4daf8f))

- Trouver la wheel reparee avec ls au lieu du glob
  ([`a1cfd75`](https://github.com/dorel14/libembedding/commit/a1cfd75b16538d4b51d2475048ecc53987a8f46b))

- Uploader seulement les wheels reparees (manylinux/macosx/win)
  ([`59b1356`](https://github.com/dorel14/libembedding/commit/59b1356ca46e2518b4e78c0be1f6610ca7b9e3dd))

### Continuous Integration

- Epingler pydantic v1 pour compatibilite semantic-release 9.4.0
  ([`407362f`](https://github.com/dorel14/libembedding/commit/407362f495bea063a0f89c26e5c57262e08527a9))

- Epingler semantic-release a 9.4.0 (stable)
  ([`821a08b`](https://github.com/dorel14/libembedding/commit/821a08bf98a2d843a061f22790661f4512a7b483))

- Installer patchelf 0.18 depuis les sources pour auditwheel
  ([`b64ea2e`](https://github.com/dorel14/libembedding/commit/b64ea2e216a98e86cc37fcb56787cdcfbe57b1b8))

- Passer a semantic-release 10.1.0 (pydantic v2 compatible)
  ([`b9c80ab`](https://github.com/dorel14/libembedding/commit/b9c80ab461949e1600c2277b7200864fe679c0f0))

- Revenir a un job release simple sans semantic-release (version depuis pyproject.toml + PYPI_TOKEN)
  ([`8f287ea`](https://github.com/dorel14/libembedding/commit/8f287eaf0c4c94986215f508126a649c430d1967))

- Revenir au workflow release simple qui marche + bump version 1.0.1
  ([`752adda`](https://github.com/dorel14/libembedding/commit/752addaa3ebb5131334d8ddde112d1cd403d6609))

- Semantic-release 9.4.0 avec pydantic v1 + PYPI_TOKEN pour publication
  ([`ce93a23`](https://github.com/dorel14/libembedding/commit/ce93a23105932eb6f2e3ee46355f0b3f0f6e6aed))

- Utiliser latest semantic-release pour compat pydantic v2
  ([`7ab04bd`](https://github.com/dorel14/libembedding/commit/7ab04bd2eda2560f3d61150916d26aae746d0a4c))


## v1.0.0 (2026-08-25)

### Continuous Integration

- Build_command doit etre une chaine, utiliser un no-op
  ([`efaa9e7`](https://github.com/dorel14/libembedding/commit/efaa9e7032f10f9029c6629dd87a0664c49ddf17))

- Desactiver le build dans semantic-release (wheels rebuilt separement)
  ([`bd7c364`](https://github.com/dorel14/libembedding/commit/bd7c364d580be0a91ada71ca51733c497ffb57b7))

- Revenir a semantic-release avec PYPI_TOKEN et rebuild des wheels apres le bump de version
  ([`eb0ac57`](https://github.com/dorel14/libembedding/commit/eb0ac57b42a18f93ecf813b9ab7dd9d2027012fc))


## v0.1.0 (2026-08-25)

### Bug Fixes

- Forcer la decouverte du package libembedding dans le wheel
  ([`5b99d0e`](https://github.com/dorel14/libembedding/commit/5b99d0e38b23edb287cc43d1daee79a6ed057aa3))

- Ne pas declarencher build.yml sur push vers main pour eviter les doublons avec release.yml
  ([`240833f`](https://github.com/dorel14/libembedding/commit/240833f4e911dead32de6a82ae1f8510826ce17d))

- Retirer la declaration duplicate de packages dans pyproject.toml
  ([`94c9729`](https://github.com/dorel14/libembedding/commit/94c9729d4d750f611a5da91876e91aff31792ca6))

- Vendor curl DLL on Windows and rename PyPI package to libembedding-ng
  ([`02c65ee`](https://github.com/dorel14/libembedding/commit/02c65eeed15c6e6c88f51f4b22793439cbd3e3ea))

- python/pyproject.toml: rename PyPI package from libembedding to libembedding-ng - CMakeLists.txt:
  make bundled third_party paths overridable via cache variables (CURL_INCLUDE_DIR, CURL_LIBRARY,
  ONNXRuntime_INCLUDE_DIR, ONNXRuntime_LIBRARY) so CI can use vcpkg's MSVC-compatible curl instead
  of the MinGW .dll.a - build-wheel.yml: on Windows, install curl via vcpkg, override curl paths to
  vcpkg's MSVC-compatible import lib, and vendor ALL third-party DLLs from vcpkg's bin (libcurl.dll,
  zlib.dll, etc.) into the wheel so it's self-contained

### Build System

- Corriger la localisation des DLL ONNX Runtime et simplifier la découverte CMake
  ([`3ccdb38`](https://github.com/dorel14/libembedding/commit/3ccdb38748cab512d0b6c447be5b0dbdbbe9fbe4))

Les DLL Windows sont désormais copiés depuis le répertoire `bin` au lieu de `lib`, correspondant à
  la structure des archives officielles ONNX Runtime. Le workflow télécharge désormais l'archive
  ONNX Runtime pour Windows au lieu d'utiliser le répertoire `third_party` empaqueté.

Le module CMake `FindONNXRuntime` inclut un fallback pour les bibliothèques partagées versionnées
  (par ex. `libonnxruntime.so.1.20.1`), et `CMakeLists.txt` utilise désormais exclusivement ce
  module personnalisé en supprimant la logique de détection via la configuration officielle.

- Documenter l'incompatibilité LTCG avec WINDOWS_EXPORT_ALL_SYMBOLS
  ([`a2be5f5`](https://github.com/dorel14/libembedding/commit/a2be5f5caadbe650258969a8eb2cc909b9c53e5b))

Une note explicative a été ajoutée dans la section des optimisations CPU du fichier CMakeLists.txt
  pour justifier l'absence des options `/GL` (optimisation whole-program) et `/LTCG` sous MSVC. Ces
  dernières sont volontairement omises car elles sont incompatibles avec
  `WINDOWS_EXPORT_ALL_SYMBOLS`, qui nécessite que CMake puisse analyser les fichiers objets afin de
  générer automatiquement le fichier définition d'exportation. Une alternative est proposée :
  fournir un fichier `.def` explicite pour activer LTCG.

### Chores

- Add python-semantic-release for automated versioning
  ([`2106fd7`](https://github.com/dorel14/libembedding/commit/2106fd7944ed371a229cd5460ab5ec17bb58d6c3))

- pyproject.toml (root): semantic-release config reading version from python/pyproject.toml, angular
  commit parser, major_on_zero for 0.x, changelog generation, GitHub release creation (no direct
  PyPI publish) - .github/workflows/semantic-release.yml: runs on push to main, uses PAT (not
  GITHUB_TOKEN) to trigger downstream publish.yml via release event -
  python/src/libembedding/__init__.py: read version from importlib.metadata instead of hardcoding to
  prevent drift from pyproject.toml

- Remove unnecessary debug symbols and unused static libraries from third_party
  ([`1a8426d`](https://github.com/dorel14/libembedding/commit/1a8426dc4a723d6d1a46de81f3ae3ad5ddea5587))

- Remove onnxruntime.pdb (389 MB debug symbols) - Remove onnxruntime_providers_shared.pdb (0.4 MB
  debug symbols) - Remove 13 unused .a static libraries from curl/lib (not referenced in
  CMakeLists.txt; only libcurl.dll.a is used for linking on Windows) - *.pdb already added to
  .gitignore in feat(build) commit

### Continuous Integration

- Add GitHub Actions workflows for building and publishing Python wheels
  ([`0b35a4d`](https://github.com/dorel14/libembedding/commit/0b35a4d5b6300f58088fcefe68069f13bfe80a14))

- build-wheel.yml: reusable workflow that builds a wheel for one platform (Linux/macOS/Windows),
  vendors ONNX Runtime via auditwheel/delocate, runs tests, and uploads the wheel as an artifact -
  build.yml: CI workflow triggered on push/PR to main, builds wheels on ubuntu-22.04, macos-14,
  macos-13, windows-2022 - publish.yml: release workflow triggered on GitHub release, builds all
  wheels and publishes to PyPI via OIDC trusted publishing - python/setup.py: minimal setup.py to
  force platform-specific wheel tags (setuptools treats .so/.dll as data files by default, producing
  py3-none-any)

- Ajouter initial_version et remettre la version a 0.1.0 pour bootstrap
  ([`35c0a90`](https://github.com/dorel14/libembedding/commit/35c0a90a9190302d4cf97099bcfdc3ef4330210f))

- Ajouter shell:bash sur l'etape de listage des roues Windows
  ([`35a8275`](https://github.com/dorel14/libembedding/commit/35a82750ee9cfbf56d865b31a6b30e2ad938111b))

- Ajouter shell:bash sur l'etape de normalisation des roues Windows
  ([`34390a4`](https://github.com/dorel14/libembedding/commit/34390a4e5215dd38c5b690d07e8535b11981c3ca))

- Corriger auditwheel/delocate et la detection ONNX Runtime sur Windows
  ([`ca5a95a`](https://github.com/dorel14/libembedding/commit/ca5a95a02245a3e534e144de01bb5eb86ec2be44))

- Corriger la configuration du workflow de publication sémantique
  ([`cd6e9e6`](https://github.com/dorel14/libembedding/commit/cd6e9e6e780c548e63b7836fac05eb41a29ade2b))

Le workflow GitHub Actions pour python-semantic-release a été mis à jour pour utiliser
  `secrets.GH_TOKEN` au lieu de `secrets.SEMANTIC_RELEASE_PAT` et ajoute une étape de configuration
  git pour le bot github-actions. Les URLs du projet dans `pyproject.toml` ont également été
  corrigées pour pointer vers le dépôt `dorel14/libembedding`.

Les fichiers README.md et python/README.md ont été mis à jour pour documenter les nouvelles
  fonctionnalités (chargement local de modèles, streaming, statistiques, fonctions de similarité) et
  les nouveaux champs d'options.

- Corriger le chemin ONNX Runtime et rétablir la version 0.1.0
  ([`f480ed4`](https://github.com/dorel14/libembedding/commit/f480ed45815b09ce603dc56f8e0636d4c01e4a99))

Le chemin `ONNXRUNTIME_ROOT` dans le workflow de construction utilise maintenant un séparateur de
  chemin cohérent (`/`), évitant les problèmes de résolution de chemin sous Windows. La version du
  projet est rétablie à `0.1.0` conformément aux versions précédentes publiées.

- Corriger le telechargement ONNX Runtime Windows et les chemins CMake
  ([`74f8fef`](https://github.com/dorel14/libembedding/commit/74f8fef6aac01ac0a85730e3be1038b43e2f62d0))

- Desactiver commit_version_number pour utiliser la version de pyproject.toml
  ([`11b4a40`](https://github.com/dorel14/libembedding/commit/11b4a4097db41c8d442008fc592f230315026660))

- Desactiver upload_to_pypi dans semantic-release et supprimer le tag v0.2.0
  ([`5f33c5f`](https://github.com/dorel14/libembedding/commit/5f33c5f3243e21bb789c3c605546e939cfdc5687))

- Exporter GH_TOKEN pour gh et garder la publication pyPI conditionnee au nouveau tag
  ([`a827657`](https://github.com/dorel14/libembedding/commit/a827657f91f73b445d7123ef342074c13e9bff2c))

- Mettre a jour la version vers 0.2.0 pour declarer la release
  ([`a665deb`](https://github.com/dorel14/libembedding/commit/a665deb3fe551309c17e870ad821aaa4016d8778))

- Rechercher les DLL ONNX Runtime recursivement sur Windows
  ([`cb36e9a`](https://github.com/dorel14/libembedding/commit/cb36e9a3e0b2620bbe9eede7cc3b25df81b86f2c))

- Remplacer semantic-release par un job de release simple (tag + github release + pypi OIDC)
  ([`492c8b9`](https://github.com/dorel14/libembedding/commit/492c8b980bfb01cbe3ff22ba3bdc87fd7d4194d7))

- Retirer macos-13 de la matrice pour debloquer la release
  ([`14b1ef1`](https://github.com/dorel14/libembedding/commit/14b1ef1c49e79b7a2c44785e5407ff9e81c30cd4))

- Utiliser libembedding.dll sur Windows pour l'export des symboles
  ([`b05b83e`](https://github.com/dorel14/libembedding/commit/b05b83e71b7aa18d32f315a155a0bf8ec2a62f43))

- Utiliser shell:bash pour l'installation du wheel sur Windows
  ([`79d251d`](https://github.com/dorel14/libembedding/commit/79d251d0ed63e701cd7873d5041ebe83fa377556))

- **release**: Transmettre les paramètres de matrice au workflow réutilisable
  ([`4869c7b`](https://github.com/dorel14/libembedding/commit/4869c7b8a21b2b6a7a09409c2ed6c52f17e18ffd))

Les paramètres `os` et `python` définis dans la matrice sont désormais explicitement passés au
  workflow réutilisable `build-wheel.yml` via la clause `with`, ce qui permet au job de construction
  de recevoir correctement les valeurs de la stratégie matricielle.

- **workflows**: Réorganiser les workflows de publication et de construction
  ([`3937f45`](https://github.com/dorel14/libembedding/commit/3937f45ea9929619f357c156316b43411b015341))

Les workflows `publish.yml` et `semantic-release.yml` ont été supprimés et remplacés par un nouveau
  workflow `release.yml` unifié. Le workflow `build-wheel.yml` a été mis à jour pour installer
  `curl` sur macOS, spécifier `shell: bash` pour l'étape de construction et rendre les arguments
  CURL optionnels dans la configuration CMake. Le fichier `CMakeLists.txt` a également été modifié
  pour supprimer l'appel à `find_package(ONNXRuntime REQUIRED)` dans la branche non-Windows, la
  dépendance ONNX Runtime étant désormais gérée différemment.

### Documentation

- Ajout de la documentation Python en français et en anglais
  ([`8fc972c`](https://github.com/dorel14/libembedding/commit/8fc972c12a9fe5ad4c7d2e6d091bb6db7069acd2))

### Features

- **api**: Ajouter le chargement local de modèles et l'introspection en temps réel
  ([`3874b62`](https://github.com/dorel14/libembedding/commit/3874b62e139f14fb0a4511a75e903c7c82d726c3))

Ajoute la capacité de charger des modèles ONNX locaux à partir de répertoires contenant model.onnx +
  tokenizer.json, pour tous les types d'embeddings (texte, sparse, image, reranker) :

- Ajoute lembed_*_create_from_path() dans l'API C pour chaque type de contexte, lisant les fichiers
  via read_file_to_string() et chargeant directement en mémoire sans téléchargement - Ajoute
  lembed_model_desc_t et lembed_*_desc() pour l'introspection en temps réel : nom, dimension,
  max_length, pooling, threads, batch_size, provider et device_id - Ajoute lembed_*_model_name() et
  lembed_*_max_length() comme accesseurs simplifiés - Ajoute le paramètre offline aux options et à
  lembed_ensure_*_model() pour forcer l'utilisation du cache uniquement - Ajoute batch_size et
  pooling aux options texte/image pour les modèles locaux sans config.json - Remplace num_threads
  par threads dans l'API Python (avec deprecation warning) et ajoute batch_size, offline, dim,
  pooling - Ajoute list_supported_models() et info() aux classes Python - Met à jour la version à
  0.2.0 et le nom du paquet PyPI en libembedding-ng

BREAKING CHANGE: le paramètre num_threads est remplacé par threads dans l'API Python TextEmbedding,
  ImageEmbedding, SparseTextEmbedding et Reranker. Le paramètre batch_size dans embed() accepte
  désormais None au lieu de 0.

- **build**: Ajouter le support Windows et les dépendances tierces
  ([`457f2fc`](https://github.com/dorel14/libembedding/commit/457f2fcf02ed2af0021642614d1497618f2c23d0))

Ajoute le support complet de Windows dans le système de build CMake : - Définit la politique CMP0144
  et NOMINMAX pour éviter les conflits min/max de l'API Windows - Configure ONNX Runtime et CURL
  comme dépendances tierces locales pour Windows, avec création de cibles importées - Passe la
  bibliothèque en mode SHARED (DLL) sous Windows pour le support Python cffi, tout en conservant le
  mode INTERFACE pour Linux/macOS - Corrige S_ISREG, ssize_t et le stockage thread-local pour la
  compatibilité MSVC dans les en-têtes de la bibliothèque - Convertit les chemins UTF-8 en UTF-16
  pour l'API CreateSession d'ONNX Runtime sous Windows

Ajoute curl et onnxruntime comme dépendances tierces dans third_party/ et ajoute la méthode
  list_supported_models() dans l'API Python TextEmbedding.

- **stats**: Ajouter les statistiques d'exécution et les fonctions de similarité
  ([`9873f18`](https://github.com/dorel14/libembedding/commit/9873f182b2bfcf618d3849300eb9214ab0fca86b))

Ajoute un système complet de statistiques en temps réel pour tous les contextes d'embedding (texte,
  sparse, image, reranker) ainsi que des fonctions de similarité vectorielle exposées dans l'API
  Python :

- Ajoute lembed_stats_t dans types.h avec texts_embedded, batches_run et avg_latency_ms - Ajoute
  lembed_*_stats() dans l'API C pour chaque type de contexte, avec chronométrage via std::chrono
  dans les fonctions embed/rerank - Ajoute lembed_version() dans config.h pour exposer la version à
  l'API C et Python - Ajoute lembed_text_embedding_embed_stream() et la méthode Python
  embed_stream() pour un traitement par lots avec rendu incrémental - Ajoute similarity.h avec
  cosine_similarity, dot_product et euclidean_distance, exposées en Python via similarity.py -
  Ajoute la classe Stats et les accesseurs stats() dans toutes les classes Python (TextEmbedding,
  SparseTextEmbedding, ImageEmbedding, Reranker) - Ajoute list_supported_models() et info() dans
  l'API Python - Ajoute des tests pour les statistiques, la similarité et max_length - Met à jour
  CMakeLists.txt pour compiler test_similarity et copier les DLLs runtime sur Windows

BREAKING CHANGE: la méthode embed() de TextEmbedding accepte désormais batch_size=None au lieu de 0
  pour utiliser la valeur par défaut du contexte.

### Breaking Changes

- **stats**: La méthode embed() de TextEmbedding accepte désormais batch_size=None au lieu de 0 pour
  utiliser la valeur par défaut du contexte.
