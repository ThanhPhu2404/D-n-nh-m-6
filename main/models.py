from django.db import models

class Vehicle(models.Model):
    track_id = models.IntegerField(unique=True)  # ID từ tracker để phân biệt xe
    license_plate = models.CharField(max_length=20, blank=True, null=True)
    vehicle_type = models.CharField(max_length=50)  # car, truck, bus, motorbike
    speed = models.FloatField(null=True, blank=True)  # tốc độ km/h

    def __str__(self):
        return f"Xe ID {self.track_id} - {self.vehicle_type} - {self.speed if self.speed else 'N/A'} km/h"


class Node(models.Model):
    name = models.CharField(max_length=100)
    lat = models.FloatField()
    lng = models.FloatField()

    def __str__(self):
        return self.name


class Edge(models.Model):
    start = models.ForeignKey(Node, related_name='edges_start', on_delete=models.CASCADE)
    end = models.ForeignKey(Node, related_name='edges_end', on_delete=models.CASCADE)
    weight = models.FloatField(help_text="Chi phí/quãng đường giữa 2 điểm")

    def __str__(self):
        return f"{self.start} -> {self.end} ({self.weight})"
