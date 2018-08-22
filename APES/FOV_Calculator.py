"FOV calculation for roguelike"
from time import time
import numpy as np

class FOV_SC(object):
    """BASED ON :http://www.roguebasin.com/index.php?title=Python_shadowcasting_implementation CODE
    Updates: 
        * Depending on Numpy array instead of string.
        * No Display since it's return stored in light attribute'
    Attributes:
        * data: numpy matrix contain the blocks(True or 1) and Empty (False or 0)
        * column: integer , max column number.
        * row: integer, max row number.
        * light: numpy matrix same size of map. contain the output where (1 is visible and 0 is not visible)
    """
    # Multipliers for transforming coordinates to other octants:
    mult = [
                [1,  0,  0, -1, -1,  0,  0,  1],
                [0,  1, -1,  0,  0, -1,  1,  0],
                [0,  1,  1,  0,  0, -1, -1,  0],
                [1,  0,  0,  1, -1,  0,  0, -1]
            ]
    def __init__(self, map):
        """initiate object from FOV_SC class
        Args:
            * map:numpy array where 0 represent clear , 1 represent block"""
        self.data = map
        self.column, self.row = map.shape[1],map.shape[0]
        self.light = np.zeros(map.shape,dtype=np.bool)

    def blocked(self, x, y):
        """Check if this coordinates refere to block
        Args:
            * x: x coordinates (or columns)
            * y: y coordinates (or Rows)
        Return: Boolean value (True blocked , False: not)"""
        return (x < 0 or y < 0
                or x >= self.column or y >= self.row
                or self.data[y,x] == 1)

    def lit(self, x, y):
        """Check if this coordinates are lighten or not.
        Args:
            * x: x coordinates (or columns)
            * y: y coordinates (or Rows)
        Return: Boolean value (True Lighten , False: not)"""
        return self.light[y,x] == 1

    def set_lit(self, x, y):
        """lighten current coordinates.
        Args:
            * x: x coordinates (or columns)
            * y: y coordinates (or Rows)"""
        if 0 <= x < self.column and 0 <= y < self.row:
            self.light[y,x] = 1

    def _cast_light(self, cx, cy, row, start, end, radius, xx, xy, yx, yy, id):
        """Recursive lightcasting function
        Args:
            * cx: x coordinates for center.
            * cy: y coordinates for center.
            * row: the row we are scanning.
            * start:
            * end: """
        if start < end:
            return
        radius_squared = radius*radius
        for j in range(row, radius+1):
            dx, dy = -j-1, -j
            blocked = False
            while dx <= 0:
                dx += 1
                # Translate the dx, dy coordinates into map coordinates:
                X, Y = cx + dx * xx + dy * xy, cy + dx * yx + dy * yy
                # l_slope and r_slope store the slopes of the left and right
                # extremities of the square we're considering:
                l_slope, r_slope = (dx-0.5)/(dy+0.5), (dx+0.5)/(dy-0.5)
                if start < r_slope:
                    continue
                elif end > l_slope:
                    break
                else:
                    # Our light beam is touching this square; light it:
                    if dx*dx + dy*dy < radius_squared:
                        self.set_lit(X, Y)
                    if blocked:
                        # we're scanning a row of blocked squares:
                        if self.blocked(X, Y):
                            new_start = r_slope
                            continue
                        else:
                            blocked = False
                            start = new_start
                    else:
                        if self.blocked(X, Y) and j < radius:
                            # This is a blocking square, start a child scan:
                            blocked = True
                            self._cast_light(cx, cy, j+1, start, l_slope,
                                             radius, xx, xy, yx, yy, id+1)
                            new_start = r_slope
            # Row is scanned; do next row unless last square was blocked:
            if blocked:
                break
    def do_fov(self, x, y):
        """Apply Shadow casting algorithm on the map from specific point prespective.
        Args:
            * x: x coordinates (or columns)
            * y: y coordinates (or Rows)
        """
        self.light[y,x]=1
        "Calculate lit squares from the given location and radius"
        for oct in range(8):
            self._cast_light(x, y, 1, 1.0, 0.0,max(self.data.shape)+2,
                             self.mult[0][oct], self.mult[1][oct],
                             self.mult[2][oct], self.mult[3][oct], 0)
