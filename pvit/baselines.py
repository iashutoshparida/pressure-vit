"""Baselines the transformer has to beat."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


class SmallCNN(nn.Module):
    """A conventional CNN with a parameter budget close to the ViT (~0.1M)."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 24), block(24, 48), block(48, 72))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1),
                                  nn.Linear(72, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def logistic_proba(x_tr, y_tr, x_te, n_classes):
    """Multinomial logistic regression on 2x2 average-pooled pixels (512 features)."""
    def pool(x):
        return x.reshape(len(x), 32, 2, 16, 2).mean(axis=(2, 4)).reshape(len(x), -1)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    clf.fit(pool(x_tr), y_tr)
    out = np.zeros((len(x_te), n_classes), dtype=np.float32)
    out[:, clf.classes_] = clf.predict_proba(pool(x_te))
    return out
