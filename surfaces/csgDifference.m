function node = csgDifference(left, right)
%CSGDIFFERENCE  Convenience constructor for a CSG difference node.

  node = csgNode('difference', left, right);
