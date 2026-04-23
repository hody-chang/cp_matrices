function node = csgIntersection(left, right)
%CSGINTERSECTION  Convenience constructor for a CSG intersection node.

  node = csgNode('intersection', left, right);
