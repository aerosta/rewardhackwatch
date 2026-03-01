import { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { cn } from '../lib/utils';

interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => React.ReactNode;
  sortable?: boolean;
  sortValue?: (row: T) => string | number;
  width?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyFn: (row: T) => string;
  emptyMessage?: string;
  onRowClick?: (row: T) => void;
  maxHeight?: string;
}

export function DataTable<T>({ columns, data, keyFn, emptyMessage = 'No data', onRowClick, maxHeight }: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  function handleSort(col: Column<T>) {
    if (!col.sortable) return;
    if (sortKey === col.key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(col.key);
      setSortDir('desc');
    }
  }

  const sorted = [...data];
  if (sortKey) {
    const col = columns.find(c => c.key === sortKey);
    if (col?.sortValue) {
      sorted.sort((a, b) => {
        const va = col.sortValue!(a);
        const vb = col.sortValue!(b);
        const cmp = va < vb ? -1 : va > vb ? 1 : 0;
        return sortDir === 'asc' ? cmp : -cmp;
      });
    }
  }

  return (
    <div className={cn('overflow-auto rounded-lg border border-border-default', maxHeight && `max-h-[${maxHeight}]`)}>
      <table className="w-full text-sm">
        <thead className="sticky top-0 z-10">
          <tr className="bg-bg-secondary border-b border-border-default">
            {columns.map(col => (
              <th
                key={col.key}
                onClick={() => handleSort(col)}
                className={cn(
                  'px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-text-muted',
                  col.sortable && 'cursor-pointer hover:text-text-primary transition-colors select-none',
                  col.width,
                )}
              >
                <div className="flex items-center gap-1">
                  {col.header}
                  {col.sortable && sortKey === col.key && (
                    sortDir === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-4 py-12 text-center text-text-muted">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sorted.map(row => (
              <tr
                key={keyFn(row)}
                onClick={() => onRowClick?.(row)}
                className={cn(
                  'border-b border-border-default/50 hover:bg-bg-elevated/50 transition-colors',
                  onRowClick && 'cursor-pointer',
                )}
              >
                {columns.map(col => (
                  <td key={col.key} className="px-4 py-3 text-text-primary">
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
