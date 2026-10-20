# Tabs

Section switching inside a page — Nodes, Shard plan, Transfers, Jobs, Gateway — with the selected tab marked by a 3px cadmium rod under its label and an optional mono count.

You provide `items` (`id`, `label`, optional `count`), the selected `value` and `onChange`; the component handles roles, `aria-selected` and arrow-key movement. Labels are one or two words in sentence case. Use at most six tabs; beyond that the content wants a sidebar. Tabs never carry status colours — a tab with a problem gets a count, and the problem is shown inside.
