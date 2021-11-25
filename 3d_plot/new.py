
import statistics

dizi = [[0,5,6,8,7,4], [9,8,7,5,4], [9,7,9,8,7,9,8,7]]

for i in range(len(dizi)):
    dizi[i].sort()

dizi.sort()
print(statistics.mean(dizi[1]))

print(dizi)