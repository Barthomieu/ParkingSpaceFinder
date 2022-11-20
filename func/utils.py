from scipy.spatial import distance


# between two points
def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


# Function to calculate K closest points
def k_closest(points, target, K):
    pts = []
    n = len(points)
    d = []

    for i in range(n):
        d.append({
            "first": distance(points[i][0], points[i][1], target[0], target[1]),
            "second": i
        })

    d = sorted(d, key=lambda l: l["first"])
    print("DEEE", d)
    print(d[i]["first"])
    for i in range(K):
        pt = []
        pt.append(points[d[i]["second"]][0])
        pt.append(points[d[i]["second"]][1])

        pts.append(pt)

    return pts


def calculate_averange_vehicle_size(coords, cars_in_area):

    vehicle_width = cars_in_area["width"].mean()  # tutaj jakaś wartość wyrażona w pixelach
    #dodac warunek ktory sprawdza orientacje samochodów względem zaznaczonych punktow

    return vehicle_width




