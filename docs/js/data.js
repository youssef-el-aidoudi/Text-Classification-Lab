/* Données du projet.
   Les scores viennent tels quels des fichiers results/*.csv.
   Les effectifs (cm) sont les comptes du jeu de test : 1115 SMS, dont 966
   messages normaux et 149 spams. Ils redonnent exactement les scores publiés.
   cm = [messages normaux laissés passer, faux spams, spams ratés, spams bloqués] */
window.LAB = {
  test: { total: 1115, ham: 966, spam: 149 },

  metrics: {
    acc: { label: "Accuracy",  short: "Accuracy",  cssVar: "--m-acc",
           help: "la part de messages bien classés, toutes classes confondues." },
    p:   { label: "Precision", short: "Precision", cssVar: "--m-p",
           help: "quand le modèle annonce « spam », a-t-il raison ? Plus le score est haut, moins il y a de fausses alertes." },
    r:   { label: "Recall",    short: "Recall",    cssVar: "--m-r",
           help: "parmi les vrais spams, combien le modèle en trouve-t-il ? Plus le score est haut, moins il en laisse passer." },
    f:   { label: "F1-score",  short: "F1",        cssVar: "--m-f",
           help: "un seul chiffre qui combine precision et recall. Il baisse si l'un des deux est faible." }
  },

  /* results/comparaison_spam.csv — scores sur la classe « spam » */
  spam: [
    { id: "nb",  name: "Naive Bayes", acc: 0.9721973094170404, p: 1.0, r: 0.7919463087248322, f: 0.8838951310861424,
      cm: [966, 0, 31, 118],
      note: "Il ne se trompe jamais quand il annonce « spam », mais il laisse passer 31 spams sur 149." },
    { id: "lr",  name: "Régression logistique", short: "Rég. logistique", acc: 0.9766816143497757, p: 0.912751677852349, r: 0.912751677852349, f: 0.912751677852349,
      cm: [953, 13, 13, 136],
      note: "Le plus équilibré : autant de fausses alertes que de spams ratés, 13 de chaque. C'est lui qui tourne dans la démo." },
    { id: "svm", name: "SVM linéaire", acc: 0.9811659192825112, p: 0.9571428571428572, r: 0.8993288590604027, f: 0.9273356401384083,
      cm: [960, 6, 15, 134],
      note: "Le meilleur compromis : il bloque 134 spams sur 149 en ne se trompant que 6 fois sur les messages normaux." },
    { id: "rf",  name: "Random Forest", acc: 0.9757847533632287, p: 1.0, r: 0.8187919463087249, f: 0.9003690036900369,
      cm: [966, 0, 27, 122],
      note: "Aucune fausse alerte non plus, mais 27 spams passent à travers." }
  ],

  /* results/partie1_mlp_imdb_results.csv */
  imdb: [
    { id: "base",  name: "MLP de base", cfg: { batch: 64, epochs: 3, lr: 0.001,  dropout: false, weighted: false },
      acc: 0.825, p: 0.8524945770065075, r: 0.786, f: 0.81789802289282,
      change: "Point de départ." },
    { id: "tuned", name: "Hyperparamètres adaptés", short: "Réglages adaptés", cfg: { batch: 32, epochs: 4, lr: 0.0005, dropout: false, weighted: false },
      acc: 0.846, p: 0.8530612244897959, r: 0.836, f: 0.8444444444444444,
      change: "Batch plus petit, learning rate plus faible, un epoch de plus." },
    { id: "drop",  name: "Avec dropout", cfg: { batch: 32, epochs: 4, lr: 0.0005, dropout: true,  weighted: false },
      acc: 0.829, p: 0.8391752577319588, r: 0.814, f: 0.8263959390862944,
      change: "Ajout du dropout (0,3)." },
    { id: "wloss", name: "Loss pondérée", cfg: { batch: 32, epochs: 4, lr: 0.0005, dropout: true,  weighted: true },
      acc: 0.839, p: 0.8466257668711656, r: 0.828, f: 0.8372093023255814,
      change: "Ajout de la loss pondérée." }
  ],

  /* results/exemples_spam_bien_mal_classes.csv — prédictions du meilleur modèle (SVM) */
  examples: [
    { text: "No need to buy lunch for me.. I eat maggi mee..", truth: "ham", model: "ham" },
    { text: "Dear Voucher Holder 2 claim your 1st class airport lounge passes when using Your holiday voucher call 08704439680. When booking quote 1st class x 2", truth: "spam", model: "ham" },
    { text: "Ok im not sure what time i finish tomorrow but i wanna spend the evening with you cos that would be vewy vewy lubly! Love me xxx", truth: "ham", model: "ham" },
    { text: "Alright i have a new goal now", truth: "ham", model: "spam" },
    { text: "Waiting in e car 4 my mum lor. U leh? Reach home already?", truth: "ham", model: "ham" }
  ],

  /* Messages de départ pour la démo (en anglais : le modèle a appris sur des SMS anglais) */
  demo: [
    { tag: "Un gain à réclamer", text: "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word CLAIM to 81010" },
    { tag: "Un vrai message",    text: "Sorry I missed your call, I was in the shower. Call me back when you can" },
    { tag: "Une offre mobile",   text: "Free entry to win a brand new mobile phone. Text WIN to 80086 now, standard rates apply" },
    { tag: "Un cas limite",      text: "Hey, free tonight? I won two tickets for the game, call me" }
  ]
};
