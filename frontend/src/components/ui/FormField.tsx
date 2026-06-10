import React from "react";

export interface FieldDef {
  key: string;
  label: string;
  type: "text" | "number" | "select";
  options?: string[];
}

interface FormFieldProps {
  field: FieldDef;
  value: string | number | null;
  onChange: (key: string, value: string | number) => void;
}

export const FormField: React.FC<FormFieldProps> = ({ field, value, onChange }) => {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={`field-${field.key}`} className="text-xs uppercase tracking-wider text-white/50 font-sans">
        {field.label}
      </label>
      {field.type === "select" ? (
        <select
          id={`field-${field.key}`}
          value={(value as string) ?? ""}
          onChange={(e) => onChange(field.key, e.target.value)}
          className="w-full bg-transparent border-b border-white/20 text-white text-sm py-2.5 appearance-none cursor-pointer hover:border-white focus:border-white transition-colors outline-none rounded-none font-sans"
        >
          <option value="" className="bg-black text-white">—</option>
          {field.options?.map((opt) => (
            <option key={opt} value={opt} className="bg-black text-white">
              {opt}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={`field-${field.key}`}
          type="number"
          value={value !== null ? String(value) : ""}
          onChange={(e) => onChange(field.key, e.target.value === "" ? "" : Number(e.target.value))}
          placeholder="0"
          className="w-full bg-transparent border-b border-white/20 text-white text-sm py-2.5 tabular-nums hover:border-white focus:border-white transition-colors outline-none rounded-none font-sans"
        />
      )}
    </div>
  );
};
