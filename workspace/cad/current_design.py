from build123d import *

with BuildPart() as p:
    with BuildSketch() as s:
        # Main profile: handle + pry head as one continuous outline
        with BuildLine() as outline:
            # Start at grip end (left)
            l1 = Line((-70, -15), (-70, 15))                    # left vertical
            l2 = RadiusArc((-70, 15), (-55, 19), 10)            # top-left fillet
            l3 = Line((-55, 19), (55, 19))                      # top handle edge
            l4 = RadiusArc((55, 19), (70, 15), 10)              # transition to head
            l5 = RadiusArc((70, 15), (88, 0), 18)               # curved pry head (top)
            l6 = RadiusArc((88, 0), (70, -15), 18)              # curved pry head (bottom)
            l7 = Line((70, -15), (-55, -19))                    # bottom handle edge
            l8 = RadiusArc((-55, -19), (-70, -15), 10)          # bottom-left fillet
        make_face()                                             # creates the solid profile
        
        # Under-cap notch (subtract from bottom edge of head)
        with Locations((70, -15)):
            Rectangle(18, 12, mode=Mode.SUBTRACT)               # notch grabs bottle cap
        
        # Lanyard hole near grip end
        with Locations((-55, 0)):
            Circle(3, mode=Mode.SUBTRACT)                       # 6 mm diameter hole
    
    extrude(amount=4)                                           # 4 mm thick titanium
    
    # Fillet all edges for comfort and to remove sharp corners
    fillet(p.edges(), radius=1)

result_part = p.part
export_stl(result_part, 'output.stl')
export_step(result_part, 'output.step')