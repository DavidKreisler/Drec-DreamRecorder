def aggregate_metric_map(names_to_tuples):
  """Aggregates the metric names to tuple dictionary.
  This function is useful for pairing metric names with their associated value
  and update ops when the list of metrics is long. For example:
  python
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        'Mean Absolute Error': new_slim.metrics.streaming_mean_absolute_error(
            predictions, labels, weights),
        'Mean Relative Error': new_slim.metrics.streaming_mean_relative_error(
            predictions, labels, labels, weights),
        'RMSE Linear': new_slim.metrics.streaming_root_mean_squared_error(
            predictions, labels, weights),
        'RMSE Log': new_slim.metrics.streaming_root_mean_squared_error(
            predictions, labels, weights),
    })

  Args:
    names_to_tuples: a map of metric names to tuples, each of which contain the
      pair of (value_tensor, update_op) from a streaming metric.
  Returns:
    A dictionary from metric names to value ops and a dictionary from metric
    names to update ops.
  """
  metric_names = names_to_tuples.keys()
  value_ops, update_ops = zip(*names_to_tuples.values())
  return dict(zip(metric_names, value_ops)), dict(zip(metric_names, update_ops))