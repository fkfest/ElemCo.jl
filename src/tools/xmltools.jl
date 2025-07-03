"""
    xspyder(xtag::AbstractString, node::Node)

Recursively search for nodes matching the xml tag `xtag` in the XML `node`.
Returns a vector of nodes that match the tag.
If `xtag` is an empty string, it returns the node itself.
"""
function xspyder(xtag::AbstractString, node::Node)
  matches = Node[]
  if xtag == ""
    push!(matches, node)
  else
    for child in children(node)
      if xtag == "*" || tag(child) == xtag
        push!(matches, child)
      end
      append!(matches, xspyder(xtag, child))
    end
  end
  return matches
end


"""
    xpath(xpath::AbstractString, node::Node)

Search for nodes matching the XPath `xpath` in the XML `node`.
Returns a vector of nodes that match the XPath expression.
If `xpath` is an empty string, it returns the node itself.

Supports basic XPath 1.0 syntax:
- `/tag1/tag2` - absolute path from root (here: root == current node)
- `tag1/tag2` - relative path from current node  
- `//tag` - descendant-or-self axis (recursive search)
- `/` - root node (here: root == current node)
- `*` - any element node
- `tag[@attr=value]` - elements with specific attribute values

Examples:
- `/node/child` - child elements named 'child' under the root 'node'
- `//item` - all 'item' elements anywhere in the tree
- `parent/child` - 'child' elements that are direct children of 'parent'
- `//variable[@name="x"]` - variable elements with name attribute equal to "x"
"""
function xpath(xpath::AbstractString, node::Node)
  if xpath == ""
    return [node]
  end
  
  # Handle root node selection
  if xpath == "/"
    # Find the root node by traversing up
    root = node
    # while XML.depth(root) !== 0
    #   root = XML.parent(root)
    # end
    return [root]
  end
  
  # Determine if this is an absolute path (starts with /)
  is_absolute = startswith(xpath, "/")
  if is_absolute
    xpath = xpath[2:end]  # Remove leading slash
    if startswith(xpath, "/")
      is_absolute = false  # If it starts with another slash, it's not absolute
      if startswith(xpath, "//")
        is_absolute = true  # If it starts with ///, it's absolute
        xpath = xpath[2:end]  # Remove double slash for descendant-or-self axis
      end
    end
  end
  
  # Start from root if absolute path, otherwise from current node
  if is_absolute
    root = node
    # while XML.depth(root) !== 0
    #   root = XML.parent(root)
    # end
    current_nodes = [root]
  else
    current_nodes = [node]
  end
  
  # Handle empty path after removing leading slash
  if xpath == ""
    return current_nodes
  end
  
  # Split path into steps, handling descendant-or-self axis (//)
  steps = String[]
  parts = split(xpath, "/")
  
  i = 1
  while i <= length(parts)
    if parts[i] == ""
      # Double slash (//) - descendant-or-self axis
      if i < length(parts)
        push!(steps, "//" * parts[i+1])
        i += 1
      end
    else
      push!(steps, parts[i])
    end
    i += 1
  end
  # Process each step
  for step in steps
    next_nodes = Node[]
    if startswith(step, "//")
      # Descendant-or-self axis - recursive search
      descendant_step = step[3:end]
      (target_tag, predicates) = parse_step_with_predicate(descendant_step)
      
      for current_node in current_nodes
        # Check if current node matches
        if (target_tag == "*" || target_tag == tag(current_node)) && 
           matches_predicates(current_node, predicates)
          push!(next_nodes, current_node)
        end
        
        # Recursively search descendants
        descendants = xspyder(target_tag, current_node)
        for desc in descendants
          if matches_predicates(desc, predicates)
            push!(next_nodes, desc)
          end
        end
      end
    else
      # Child axis - direct children only
      (target_tag, predicates) = parse_step_with_predicate(step)
      
      for current_node in current_nodes
        for child in children(current_node)
          if (target_tag == "*" || target_tag == tag(child)) && 
             matches_predicates(child, predicates)
            push!(next_nodes, child)
          end
        end
      end
    end
    current_nodes = next_nodes
  end
  return current_nodes
end

"""
    parse_step_with_predicate(step::AbstractString)

Parse a step that may contain predicates like "tag[@attr=value]".
Returns (tag_name, predicates) where predicates is a vector of predicate strings.
"""
function parse_step_with_predicate(step::AbstractString)
  # Check if step contains predicates
  if !occursin('[', step)
    return (step, String[])
  end
  
  # Find the tag name (before the first '[')
  bracket_pos = findfirst('[', step)
  tag_name = step[1:bracket_pos-1]
  
  # Extract predicates between brackets
  predicates = String[]
  remaining = step[bracket_pos:end]
  
  while occursin('[', remaining) && occursin(']', remaining)
    start_bracket = findfirst('[', remaining)
    end_bracket = findfirst(']', remaining)
    if start_bracket !== nothing && end_bracket !== nothing && end_bracket > start_bracket
      predicate = remaining[start_bracket+1:end_bracket-1]
      push!(predicates, predicate)
      remaining = remaining[end_bracket+1:end]
    else
      break
    end
  end
  
  return (tag_name, predicates)
end

"""
    evaluate_predicate(predicate::AbstractString, node::Node)

Evaluate a predicate against a node. Currently supports:
- @attr=value
"""
function evaluate_predicate(predicate::AbstractString, node::Node)
  # Handle attribute predicates like @name=value or @name=$var
  if occursin('=', predicate)
    parts = split(predicate, '=', limit=2)
    if length(parts) == 2
      attr_part = strip(parts[1])
      value_part = strip(parts[2])
      
      # Remove quotes from value if present
      if (startswith(value_part, '"') && endswith(value_part, '"')) || 
         (startswith(value_part, '\'') && endswith(value_part, '\''))
        value_part = value_part[2:end-1]
      end
      
      # Handle attribute access
      if startswith(attr_part, '@')
        attr_name = attr_part[2:end]
        # Get attribute value from the node
        attrs = attributes(node)
        attr = get(attrs, attr_name, nothing)
        if !isnothing(attr)
          # Check if the attribute value matches the expected value
          return attr == value_part
        else
          return false  # Attribute not found
        end
      end
    end
  end
  # Default: predicate not supported or doesn't match
  return false
end

"""
    matches_predicates(node::Node, predicates::Vector{String})

Check if a node matches all given predicates.
"""
function matches_predicates(node::Node, predicates::Vector{String})
  for predicate in predicates
    if !evaluate_predicate(predicate, node)
      return false
    end
  end
  return true
end