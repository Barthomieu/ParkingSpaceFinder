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
    for i in range(K):
        pt = []
        pt.append(points[d[i]["second"]][0])
        pt.append(points[d[i]["second"]][1])
        pt.append(d[i]["first"])

        pts.append(pt)

    return pts


def calculate_averange_vehicle_size(coords, cars_in_area):

    vehicle_width = cars_in_area["width"].mean()  # tutaj jakaś wartość wyrażona w pixelach
    #dodac warunek ktory sprawdza orientacje samochodów względem zaznaczonych punktow

    return vehicle_width


def area_points(p1,p2):
    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)
    mid= midpoint(p1,p2)
    return [p1,p2,dist,mid]


def midpoint(p1, p2):
    return (round((p1[0] + p2[0]) / 2), round((p1[1] + p2[1]) / 2))


def find_shorter_side(coord):
    side_length = []
    for i in range(len(coord)):
        side_length.append(area_points(coord[i-1], coord[i]))
    side_length.sort(key=lambda x: x[2]) # sortowanie po po długości

    return side_length[0:2]

