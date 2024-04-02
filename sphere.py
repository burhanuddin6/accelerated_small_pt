import taichi as ti

@ti.dataclass
class Sphere():
    r: ti.f64
    p: ti.math.vec3
    e: ti.math.vec3
    f: ti.math.vec3
    reflection_t: int

    @ti.func
    def intersect(self, ray):
        # (o + t*d - p) . (o + t*d - p) - r*r = 0
        # <=> (d . d) * t^2 + 2 * d . (o - p) * t + (o - p) . (o - p) - r*r = 0
        # 
        # Discriminant check
        # (2 * d . (o - p))^2 - 4 * (d . d) * ((o - p) . (o - p) - r*r) <? 0
        # <=> (d . (o - p))^2 - (d . d) * ((o - p) . (o - p) - r*r) <? 0
        # <=> (d . op)^2 - 1 * (op . op - r*r) <? 0
        # <=> b^2 - (op . op) + r*r <? 0
        # <=> D <? 0
        #
        # Solutions
        # t = (- 2 * d . (o - p) +- 2 * sqrt(D)) / (2 * (d . d))
        # <=> t = dop +- sqrt(D)

        op = self.p - ray.o
        dop = ray.d.dot(op)
        D = dop * dop - op.dot(op) + self.r * self.r

        if D < 0:
            return False

        sqrtD = ti.math.sqrt(D)

        tmin = dop - sqrtD
        if (ray.tmin < tmin and tmin < ray.tmax):
            ray.tmax = tmin
            return True

        tmax = dop + sqrtD
        if (ray.tmin < tmax and tmax < ray.tmax):
            ray.tmax = tmax
            return True
        
        return False
