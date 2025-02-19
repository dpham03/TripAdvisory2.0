from django.db import models
from constants import Config
class EmbeddingConfig(models.Model):
    alpha = models.FloatField(default=Config.ALPHA_DEFAULT)
    beta = models.FloatField(default=Config.BETA_DEFAULT)
    
    def save(self, *args, **kwargs):
        if self.alpha + self.beta != 1:
            raise ValueError("Alpha and Beta must sum to 1.")
        super().save(*args, **kwargs)