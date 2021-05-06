def prueba():
    print('hello world')
    out=100
    return out

def canyon(x0,v,alpha):
    import math
    radianes=math.radians(alpha)
    vx=math.cos(radianes)*v
    vy=math.sin(radianes)*v
    t=vy*2/9.8
    x=x0+vx*t
    
    return x
class canyon(object):
    # Esta clase calcula tiro parabólico.
    def __init__ (self,x0):
        self.x0=x0
        
        
    def calcular(self,x0,v,alpha):
        import math
        self.x0=x0
        self.v=x0
        self.alpha=x0
        radianes=math.radians(alpha)
        vx=math.cos(radianes)*v
        vy=math.sin(radianes)*v
        t=vy*2/9.8
        x=x0+vx*t
        print('El punto de caída es',x)
        self.vx=vx
        self.vy=vy
        self.t=t