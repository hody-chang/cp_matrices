function node = csgUnion(left, right)
%CSGUNION  Convenience constructor for a CSG union node.

  node = csgNode('union', left, right);
